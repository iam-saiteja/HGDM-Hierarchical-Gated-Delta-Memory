import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
from kernel_nitro import fused_nitro_scan

# =============================================================================
# 1. THE CONFIGURATION (120M Baseline)
# =============================================================================
@dataclass
class HGDMConfig:
    d_model: int = 768         
    n_layers: int = 12         
    n_heads: int = 12          
    d_k: int = 64              
    d_v: int = 64              
    d_ff: int = 3072           
    vocab_size: int = 256      
    use_variable_delta_t: bool = True   # [STEP-01] Time-based model: content-driven Δt decay (CfC/Mamba)
    use_state_fusion: bool = False      # Feature 6: Cross-Layer State Fusion (State Highways)
    max_position_embeddings: int = 2048 # Bug 2 Fix: Configurable positional embedding size (saves 201MB VRAM by default)

# =============================================================================
# 2. THE COMPONENTS
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class MultiHeadGatedDelta(nn.Module):
    def __init__(self, config: HGDMConfig, force_sequential=False):
        super().__init__()
        self.config = config
        self.H = config.n_heads
        self.d_k = config.d_k
        self.d_v = config.d_v
        self.force_sequential = force_sequential
        
        self.W_q = nn.Linear(config.d_model, self.H * self.d_k, bias=False)
        self.W_k = nn.Linear(config.d_model, self.H * self.d_k, bias=False)
        self.W_v = nn.Linear(config.d_model, self.H * self.d_v, bias=False)
        
        # [STEP-04] Asymmetric Decay Init: cortical half/half split
        # Fast heads (h < H//2): τ = 4*(h+1)  → short timescales [4, 8, 12, ...]
        # Slow heads (h >= H//2): τ = 200*(h-H//2+1) → long timescales [200, 400, 600, ...]
        # Replaces cyclic base_taus which arbitrarily mixed timescales across heads.
        if getattr(config, "use_variable_delta_t", False):
            initial_lambdas = []
            H_half = self.H // 2
            for h in range(self.H):
                if h < H_half:
                    tau = 4.0 * (h + 1)           # fast: 4, 8, 12, 16, 24, ...
                else:
                    tau = 200.0 * (h - H_half + 1) # slow: 200, 400, 600, 800, ...
                initial_lambdas.append(1.0 / tau)
            self.W_lambda = nn.Parameter(torch.log(torch.tensor(initial_lambdas, dtype=torch.float32)))
            self.W_delta = nn.Linear(config.d_model, self.H, bias=True)
        else:
            self.W_alpha = nn.Linear(config.d_model, self.H)
            
        self.W_beta  = nn.Linear(config.d_model, self.H)
        self.W_out_gate = nn.Linear(config.d_model, self.H * self.d_v)
        self.W_o = nn.Linear(self.H * self.d_v, config.d_model, bias=False)
        
        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            if getattr(self.config, "use_variable_delta_t", False):
                # Initialize W_delta weight to 0.0 for initial input-independence
                self.W_delta.weight.zero_()
                # Bug 4 Fix: Initialize W_delta.bias to 0.5413 to get delta_t = 1.0 initially, preserving timescales at step 0
                self.W_delta.bias.fill_(0.5413)
            else:
                base_taus = [4.0, 30.0, 200.0, 1200.0, 8000.0] 
                biases = []
                for h in range(self.H):
                    tau = base_taus[h % len(base_taus)] 
                    alpha_target = math.exp(-1.0 / tau)
                    bias_val = math.log(alpha_target / (1.0 - alpha_target + 1e-8))
                    biases.append(bias_val)
                self.W_alpha.bias.copy_(torch.tensor(biases))
                torch.nn.init.normal_(self.W_alpha.weight, mean=0.0, std=0.02) 

            # [STEP-02] QK-Norm makes W_q/W_k scale irrelevant (output is always unit-norm)
            # Removed: self.W_q.weight.data *= 0.1 and self.W_k.weight.data *= 0.1
            self.W_beta.weight.zero_()
            self.W_beta.bias.fill_(-1.0) 
            self.W_out_gate.weight.zero_()
            self.W_out_gate.bias.zero_()

    def forward(self, x, state=None):
        B, T, _ = x.shape
        # [STEP-02] QK-Norm: L2-normalize q and k to unit sphere before state update.
        q = F.normalize(self.W_q(x).view(B, T, self.H, self.d_k), dim=-1)
        k = F.normalize(self.W_k(x).view(B, T, self.H, self.d_k), dim=-1)
        v = self.W_v(x).view(B, T, self.H, self.d_v)
        
        # Variable-Delta-t ODE Decay
        if getattr(self.config, "use_variable_delta_t", False):
            delta_t = F.softplus(self.W_delta(x)) + 1e-3
            lambdas = torch.exp(self.W_lambda)
            alpha = torch.exp(-delta_t * lambdas[None, None, :])
        else:
            alpha = torch.sigmoid(self.W_alpha(x))
            
        # [STEP-07] Sparse Write Gate: shifted ReLU — only strong signals write to state
        # beta < 0.1 → exactly 0 (write blocked), beta > 0.1 → rescaled to [0,1]
        # Biological analogy: LTP threshold — weak signals are filtered, only strong ones consolidate
        _beta_raw = torch.sigmoid(self.W_beta(x))
        beta      = F.relu(_beta_raw - 0.1) / 0.9
        out_gate = torch.sigmoid(self.W_out_gate(x)).view(B, T, self.H, self.d_v)

        # [STEP-05] Unpack (S, n) state tuple, or init both to None for fresh start
        if state is not None:
            S_prev, n_prev = state
        else:
            S_prev, n_prev = None, None

        if not self.force_sequential and fused_nitro_scan is not None and q.is_cuda:
            # FAST PATH: Triton kernel handles the expensive S recurrence
            out, S = fused_nitro_scan(q, k, v, alpha, beta, S_prev)

            # [STEP-05] n_t recurrence in PyTorch — cheap O(T*H*d_k), no kernel change needed
            # n_t = alpha_t * n_{t-1} + beta_t * k_t  (tracks write accumulation per key dim)
            n = n_prev if n_prev is not None else torch.zeros(B, self.H, self.d_k, device=x.device, dtype=x.dtype)
            n_list = []
            for t in range(T):
                n = alpha[:, t, :, None] * n + beta[:, t, :, None] * k[:, t]
                n_list.append(n)
            n_stack = torch.stack(n_list, dim=1)  # [B, T, H, d_k]

            # [STEP-05] Normalize: out = (q @ S) / max(||n||_inf, 1)
            n_inf  = n_stack.abs().max(dim=-1)[0]                # [B, T, H]
            denom  = torch.clamp(n_inf, min=1.0).unsqueeze(-1)   # [B, T, H, 1]
            # [STEP-06] Epistemic gate: confidence = tanh(||n||_2) per head
            # Fresh state (n=0): conf=0 → model is silent (no hallucination at start)
            # Rich state (n large): conf→1 → full output
            conf   = torch.tanh(n_stack.norm(dim=-1)).unsqueeze(-1)  # [B, T, H, 1]
            out    = (out / denom) * out_gate * conf
            out    = out.reshape(B, T, -1)
            return self.W_o(out), (S, n)

        else:
            # SEQUENTIAL PATH: Pure PyTorch fallback
            S = S_prev if S_prev is not None else torch.zeros(B, self.H, self.d_k, self.d_v, device=x.device, dtype=x.dtype)
            n = n_prev if n_prev is not None else torch.zeros(B, self.H, self.d_k, device=x.device, dtype=x.dtype)
            outputs = []
            for t in range(T):
                delta = torch.einsum('bhk,bhd->bhkd', k[:, t], v[:, t])
                S = alpha[:, t, :, None, None] * S + beta[:, t, :, None, None] * delta
                # [STEP-05] n_t update + normalize
                n = alpha[:, t, :, None] * n + beta[:, t, :, None] * k[:, t]
                n_inf  = n.abs().max(dim=-1)[0]                    # [B, H]
                denom  = torch.clamp(n_inf, min=1.0).unsqueeze(-1) # [B, H, 1]
                # [STEP-06] Epistemic gate per head
                conf   = torch.tanh(n.norm(dim=-1)).unsqueeze(-1)   # [B, H, 1]
                out_t  = torch.einsum('bhkd,bhk->bhd', S, q[:, t]) / denom * out_gate[:, t] * conf
                outputs.append(out_t)
            out = torch.stack(outputs, dim=1).reshape(B, T, -1)
            return self.W_o(out), (S, n)

class HGDMLayer(nn.Module):
    def __init__(self, config: HGDMConfig, layer_idx: int, force_sequential=False):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.mixer = MultiHeadGatedDelta(config, force_sequential=force_sequential)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff)
        
    def forward(self, x, state=None):
        m_out, new_mixer_state = self.mixer(self.norm1(x), state=state)
        x = x + m_out
        x = x + self.ffn(self.norm2(x))
        return x, new_mixer_state

# Advanced Feature 6: Cross-Layer State Fusion (State Highways)
class CrossLayerStateFusion(nn.Module):
    def __init__(self, config: HGDMConfig):
        super().__init__()
        self.config = config
        # Bug 1 Fix: Initialize fusion_gate to -4.0 so sigmoid(-4) ≈ 0.018 (starts close to 0)
        self.fusion_gate = nn.Parameter(torch.full((config.n_layers, config.n_heads), -4.0))

    def fuse(self, S_current, S_prev_layer, layer_idx):
        gate = torch.sigmoid(self.fusion_gate[layer_idx])  # [H]
        gate = gate[None, :, None, None]                   # broadcast to [1, H, 1, 1]
        return S_current + gate * S_prev_layer             # additive state highway

class HGDMUltimate(nn.Module):
    def __init__(self, config: HGDMConfig, force_sequential=False):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Bug 2 Fix: Configure positional embedding size from configuration to save VRAM
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.max_position_embeddings, config.d_model) * 0.02
        )
        
        self.layers = nn.ModuleList([
            HGDMLayer(config, i, force_sequential=force_sequential) for i in range(config.n_layers)
        ])
        
        # Advanced Feature 6: State Fusion Highway
        if getattr(config, "use_state_fusion", False):
            self.state_fusion = CrossLayerStateFusion(config)
            
        self.norm_f = RMSNorm(config.d_model)
        self.fc_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.fc_out.weight = self.embedding.weight

    def forward(self, byte_seq, states=None, offset=0):
        B, T = byte_seq.shape
        x = self.embedding(byte_seq)
        
        # Bug 2 Wrap-around check: Allow arbitrary token offsets during generation without out-of-bounds errors
        pos_offset = offset % self.pos_embedding.shape[1]
        if pos_offset + T > self.pos_embedding.shape[1]:
            indices = torch.arange(offset, offset + T, device=byte_seq.device) % self.pos_embedding.shape[1]
            x = x + self.pos_embedding[:, indices, :]
        else:
            x = x + self.pos_embedding[:, pos_offset : pos_offset + T, :]
        
        if states is None: states = [None] * len(self.layers)
        next_states = []
        prev_state = None
        
        for i, layer in enumerate(self.layers):
            x, ns = layer(x, states[i])
            # Bug 3 Fix: Track raw unfused recurrent state to prevent cascade explosion
            unfused_ns = ns
            
            # Cross-layer state highway fusion
            if getattr(self.config, "use_state_fusion", False) and i > 0 and prev_state is not None:
                ns = self.state_fusion.fuse(ns, prev_state, i)
                
            # Cascade the unfused raw state to isolate fusion to direct adjacent connections
            prev_state = unfused_ns
            next_states.append(ns)
            
        x = self.norm_f(x)
        return self.fc_out(x), next_states

    @torch.no_grad()
    def generate(self, prompt_bytes, max_new_bytes=100, temp=0.8):
        self.eval()
        generated = prompt_bytes
        logits, states = self.forward(prompt_bytes)
        next_logit = logits[:, -1, :] / temp
        next_probs = F.softmax(next_logit, dim=-1)
        next_byte = torch.multinomial(next_probs, num_samples=1)
        generated = torch.cat([generated, next_byte], dim=1)
        
        offset = prompt_bytes.shape[1]
        for _ in range(max_new_bytes - 1):
            logits, next_states = self.forward(next_byte, states, offset=offset)
            states = next_states
            next_logit = logits[:, -1, :] / temp
            next_probs = F.softmax(next_logit, dim=-1)
            next_byte = torch.multinomial(next_probs, num_samples=1)
            generated = torch.cat([generated, next_byte], dim=1)
            offset += 1
        return generated