import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from tokenized.config import TokenizedConfig

# Attempt to import custom Triton scan kernel if available
try:
    from kernel_nitro import fused_nitro_scan, fused_nitro_scan_with_n
except (ImportError, Exception):
    fused_nitro_scan = None
    fused_nitro_scan_with_n = None


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class TokenizedMHGD(nn.Module):
    """
    Multi-Head Gated Delta recurrence for tokenized sequences.
    Features:
      1. Continuous CfC / Neural ODE decay (variable Delta t)
      2. QK-Norm on unit sphere for numerical stability
      3. State normalizer n_t + Epistemic Confidence Gating tanh(||n_t||)
      4. Sparse write gate (shifted ReLU beta)
      5. Multi-timescale decay prior (fast to slow heads)
      6. Triton Nitro parallel scan with vectorized PyTorch fallback
    """
    def __init__(self, config: TokenizedConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.H = config.n_heads
        self.d_k = config.d_k
        self.d_v = config.d_v
        self.d_model = config.d_model
        
        # Projections
        self.W_q = nn.Linear(config.d_model, self.H * self.d_k, bias=False)
        self.W_k = nn.Linear(config.d_model, self.H * self.d_k, bias=False)
        self.W_v = nn.Linear(config.d_model, self.H * self.d_v, bias=False)
        self.W_o = nn.Linear(self.H * self.d_v, config.d_model, bias=False)
        
        # Gates
        self.W_beta = nn.Linear(config.d_model, self.H, bias=True)
        self.W_out  = nn.Linear(config.d_model, self.H * self.d_v, bias=True)
        self.log_beta_scale = nn.Parameter(torch.zeros(self.H))
        
        # Continuous variable Delta t ODE decay
        if config.use_variable_delta_t:
            self.W_delta = nn.Linear(config.d_model, self.H, bias=True)
            self.W_lambda = nn.Parameter(torch.empty(self.H))
        else:
            self.W_alpha = nn.Linear(config.d_model, self.H, bias=True)
            
        self._init_weights()

    def _init_weights(self):
        # QK-Norm scale
        nn.init.normal_(self.W_q.weight, std=0.02)
        nn.init.normal_(self.W_k.weight, std=0.02)
        nn.init.normal_(self.W_v.weight, std=0.02)
        nn.init.normal_(self.W_o.weight, std=0.02)
        
        # Sparse beta gate initialization
        nn.init.zeros_(self.W_beta.weight)
        nn.init.constant_(self.W_beta.bias, 0.0)
        
        # Out gate init open
        nn.init.zeros_(self.W_out.weight)
        nn.init.constant_(self.W_out.bias, 2.0)
        
        if self.config.use_variable_delta_t:
            # Variable Delta t initialization:
            # delta_t = softplus(W_delta(x)) + 1e-3. At x=0, softplus(0.5413) ~= 1.0
            nn.init.zeros_(self.W_delta.weight)
            nn.init.constant_(self.W_delta.bias, 0.5413)
            
            # Multi-timescale decay prior calibrated for subword tokens:
            # Fast heads: tau in [2, 16] tokens (local phrases)
            # Medium heads: tau in [32, 128] tokens (sentences)
            # Slow heads: tau in [256, 2048] tokens (paragraphs/discourse)
            taus = []
            half_h = self.H // 2
            for h in range(half_h):
                taus.append(2.0 * ((16.0 / 2.0) ** (h / max(1, half_h - 1))))
            for h in range(self.H - half_h):
                taus.append(32.0 * ((2048.0 / 32.0) ** (h / max(1, self.H - half_h - 1))))
            
            lambdas = [1.0 / max(1.0, tau) for tau in taus]
            with torch.no_grad():
                self.W_lambda.copy_(torch.log(torch.tensor(lambdas, dtype=torch.float32)))
        else:
            nn.init.zeros_(self.W_alpha.weight)
            nn.init.constant_(self.W_alpha.bias, 2.0)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Args:
            x: [B, T, d_model]
            state: Optional tuple (S, n) where:
                S: [B, H, d_k, d_v] recurrent matrix state
                n: [B, H, d_k] state normalizer vector
        Returns:
            out: [B, T, d_model]
            new_state: (S, n)
        """
        B, T, _ = x.shape
        
        # 1. Project Q, K, V
        q = self.W_q(x).view(B, T, self.H, self.d_k)
        k = self.W_k(x).view(B, T, self.H, self.d_k)
        v = self.W_v(x).view(B, T, self.H, self.d_v)
        
        # QK-Norm: Project q and k to unit sphere
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)
        
        # 2. Gates
        # Output gate
        out_gate = torch.sigmoid(self.W_out(x)).view(B, T, self.H, self.d_v)
        
        # Sparse Write gate beta: Shifted ReLU for 10-20% write sparsity
        beta_raw = torch.sigmoid(self.W_beta(x))
        thresh = self.config.sparse_write_threshold
        beta = (F.relu(beta_raw - thresh) / (1.0 - thresh + 1e-8)) * torch.exp(self.log_beta_scale)[None, None, :]
        
        # Continuous Decay gate alpha: exp(-Delta_t * lambda_h)
        if self.config.use_variable_delta_t:
            delta_t = F.softplus(self.W_delta(x)) + 1e-3
            decay_rate = torch.exp(self.W_lambda)[None, None, :]
            alpha = torch.exp(-delta_t * decay_rate).clamp(min=1e-4, max=0.9999)
        else:
            alpha = torch.sigmoid(self.W_alpha(x)).clamp(min=1e-4, max=0.9999)
            
        S_prev, n_prev = state if state is not None else (None, None)
        
        # Fast Triton scan if available and training/full sequence on GPU
        if (
            fused_nitro_scan_with_n is not None 
            and x.is_cuda 
            and T > 1 
            and (T % self.config.chunk_size == 0)
        ):
            if self.config.use_epistemic_gate:
                out, S, n_stack = fused_nitro_scan_with_n(
                    q, k, v, alpha, beta, 
                    state=S_prev, 
                    initial_n=n_prev,
                    chunk_size=self.config.chunk_size
                )
                n_t = n_stack.transpose(1, 2)  # [B, H, T, dk] -> [B, T, H, dk]
                n_inf = n_t.abs().max(dim=-1)[0]
                denom = torch.clamp(n_inf, min=1.0).unsqueeze(-1)
                conf = torch.tanh(n_t.norm(dim=-1)).unsqueeze(-1)
                out = (out / denom) * out_gate * conf
                n = n_t[:, -1]
            else:
                out, S = fused_nitro_scan(
                    q, k, v, alpha, beta, 
                    state=S_prev, 
                    chunk_size=self.config.chunk_size
                )
                out = out * out_gate
                n = None
            return self.W_o(out.reshape(B, T, -1)), (S, n)
            
        # Parallel Vectorized PyTorch Scan (when starting from fresh state and T > 1)
        elif S_prev is None and T > 1:
            log_alpha = torch.log(alpha.clamp(min=1e-7)).permute(0, 2, 1)  # [B, H, T]
            cumsum_log = torch.cumsum(log_alpha, dim=-1)
            diff = cumsum_log.unsqueeze(-1) - cumsum_log.unsqueeze(-2)      # [B, H, T, T]
            
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            diff = torch.where(causal_mask, diff, torch.full_like(diff, float('-inf')))
            decay_matrix = torch.exp(diff) * beta.permute(0, 2, 1).unsqueeze(-2)  # [B, H, T, T]
            
            q_p = q.permute(0, 2, 1, 3)  # [B, H, T, d_k]
            k_p = k.permute(0, 2, 1, 3)  # [B, H, T, d_k]
            v_p = v.permute(0, 2, 1, 3)  # [B, H, T, d_v]
            
            qk = torch.matmul(q_p, k_p.transpose(-1, -2))  # [B, H, T, T]
            attn = qk * decay_matrix
            out = torch.matmul(attn, v_p).permute(0, 2, 1, 3)  # [B, T, H, d_v]
            
            # State Normalizer & Epistemic Confidence
            if self.config.use_epistemic_gate:
                n_vec = torch.matmul(decay_matrix, k_p).permute(0, 2, 1, 3)  # [B, T, H, d_k]
                n_inf = n_vec.abs().max(dim=-1, keepdim=True)[0]
                denom = torch.clamp(n_inf, min=1.0)
                conf = torch.tanh(n_vec.norm(dim=-1, keepdim=True))
                out = (out / denom) * conf
                n_last = n_vec[:, -1]
            else:
                n_last = None
                
            out = (out * out_gate).reshape(B, T, -1)
            return self.W_o(out), (None, n_last)
            
        # Step-by-Step Recurrent Mode (O(1) Memory Autoregressive Generation)
        else:
            S = S_prev if S_prev is not None else torch.zeros(
                B, self.H, self.d_k, self.d_v, device=x.device, dtype=x.dtype
            )
            if self.config.use_epistemic_gate:
                n = n_prev if n_prev is not None else torch.zeros(
                    B, self.H, self.d_k, device=x.device, dtype=x.dtype
                )
            else:
                n = None
                
            outputs = []
            for t in range(T):
                # Outer product update: delta = k_t (x) v_t
                delta = torch.einsum('bhk,bhd->bhkd', k[:, t], v[:, t])
                S = alpha[:, t, :, None, None] * S + beta[:, t, :, None, None] * delta
                
                # Associative Readout: out_t = S * q_t
                out_t = torch.einsum('bhkd,bhk->bhd', S, q[:, t])
                
                # Epistemic Confidence & Normalization
                if self.config.use_epistemic_gate:
                    n = alpha[:, t, :, None] * n + beta[:, t, :, None] * k[:, t]
                    n_inf = n.abs().max(dim=-1)[0]
                    denom = torch.clamp(n_inf, min=1.0).unsqueeze(-1)
                    conf = torch.tanh(n.norm(dim=-1)).unsqueeze(-1)
                    out_t = (out_t / denom) * conf
                    
                out_t = out_t * out_gate[:, t]
                outputs.append(out_t)
                
            out = torch.stack(outputs, dim=1).reshape(B, T, -1)
            return self.W_o(out), (S, n)


class TokenizedHGDMLayer(nn.Module):
    def __init__(self, config: TokenizedConfig, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.mixer = TokenizedMHGD(config, layer_idx)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config.d_model, config.d_ff, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        mixer_out, next_state = self.mixer(self.norm1(x), state=state)
        x = x + mixer_out
        x = x + self.mlp(self.norm2(x))
        return x, next_state


class TokenizedHGDM(nn.Module):
    """
    Subword-Tokenized HGDM Model.
    - Zero Attention / Zero KV Cache.
    - Constant O(1) Memory during generation.
    - Tied input embedding and output projection.
    """
    def __init__(self, config: TokenizedConfig):
        super().__init__()
        self.config = config
        
        # Tied Token Embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Layers
        self.layers = nn.ModuleList([
            TokenizedHGDMLayer(config, idx) for idx in range(config.n_layers)
        ])
        
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Weight tying
        
        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, Optional[torch.Tensor]]]]:
        """
        Forward pass.
        input_ids: [B, T]
        states: Optional list of states per layer
        """
        x = self.embedding(input_ids)
        
        next_states = []
        for i, layer in enumerate(self.layers):
            layer_state = states[i] if states is not None else None
            x, next_state = layer(x, state=layer_state)
            next_states.append(next_state)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, next_states

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Pure O(1) Memory Autoregressive Generation.
        Processes prompt once to prime state, then emits token-by-token
        using only the persistent recurrent state matrix (zero KV cache).
        """
        self.eval()
        B, T_prompt = prompt_ids.shape
        device = prompt_ids.device
        
        # 1. Prime recurrent states over the prompt
        _, states = self(prompt_ids)
        
        # 2. Sequential generation loop carrying ONLY states
        curr_token = prompt_ids[:, -1:]  # [B, 1]
        generated = [prompt_ids]
        
        for _ in range(max_new_tokens):
            logits, states = self(curr_token, states=states)
            logits_step = logits[:, -1, :] / max(1e-5, temperature)
            
            if top_k > 0:
                v, _ = torch.topk(logits_step, min(top_k, logits_step.size(-1)))
                logits_step[logits_step < v[:, [-1]]] = -float('inf')
                
            probs = F.softmax(logits_step, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated.append(next_token)
            curr_token = next_token
            
        return torch.cat(generated, dim=1)
