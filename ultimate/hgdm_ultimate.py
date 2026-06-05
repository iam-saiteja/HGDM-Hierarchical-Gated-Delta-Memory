import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
try:
    from kernel_nitro import fused_nitro_scan, fused_nitro_scan_with_n
except (ImportError, Exception):
    fused_nitro_scan = None
    fused_nitro_scan_with_n = None

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
    d_ff: int = 2048           
    vocab_size: int = 256      
    use_variable_delta_t: bool = True   # [STEP-01] Time-based model: content-driven Δt decay (CfC/Mamba)
    use_state_fusion: bool = False      # Feature 6: Cross-Layer State Fusion (State Highways)
    max_position_embeddings: int = 2048 # Bug 2 Fix: Configurable positional embedding size (saves 201MB VRAM by default)
    use_rope: bool = False              # [STEP-11] Rotary Position Embeddings (RoPE)
    boundary_token_ids: Tuple[int, ...] = (46, 63, 33, 10)
    n_grad_mode: str = "detached"       # "detached" saves VRAM; "exact" backprops through n.
    use_epistemic_gate: bool = True     # Use per-token epistemic confidence gating

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

class RoPEEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len, offset=0):
        total_len = offset + seq_len
        if total_len > self.cos_cached.shape[0]:
            new_max = max(total_len, self.cos_cached.shape[0] * 2)
            t = torch.arange(new_max, device=x.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(x.device))
            emb = torch.cat((freqs, freqs), dim=-1)
            # Direct reassignment updates the buffer in self._buffers without register_buffer() overhead
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
            
        cos = self.cos_cached[offset:total_len].to(x.device)
        sin = self.sin_cached[offset:total_len].to(x.device)
            
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

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
        
        # [STEP-11] Rotary Position Embeddings (RoPE)
        self.use_rope = getattr(config, "use_rope", False)
        if self.use_rope:
            self.rope_emb = RoPEEmbedding(self.d_k, max_position_embeddings=getattr(config, "max_position_embeddings", 2048))
        
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
            
        self.W_beta      = nn.Linear(config.d_model, self.H)
        # [STEP-08] Per-head write scale: exp(log_beta_scale_h) multiplies beta for head h
        # Starts at 0 → exp(0)=1 → no change at init. Learned divergence during training.
        self.log_beta_scale = nn.Parameter(torch.zeros(self.H))
        # [STEP-09] Phase Oscillator: learnable period per head
        # Fast heads: init T_cycle = 8.0, slow heads: init T_cycle = 512.0
        self.log_T_cycle = nn.Parameter(torch.cat([
            torch.log(torch.full((self.H // 2,), 8.0)),
            torch.log(torch.full((self.H - self.H // 2,), 512.0))
        ]))
        self.W_out_gate  = nn.Linear(config.d_model, self.H * self.d_v)
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

    def forward(self, x, state=None, boundary_mask=None, offset=0):
        B, T, _ = x.shape
        # [STEP-02] QK-Norm: L2-normalize q and k to unit sphere before state update.
        q = F.normalize(self.W_q(x).view(B, T, self.H, self.d_k), dim=-1, eps=1e-6)
        k = F.normalize(self.W_k(x).view(B, T, self.H, self.d_k), dim=-1, eps=1e-6)
        
        # [STEP-11] Rotary Position Embeddings (RoPE)
        if getattr(self.config, "use_rope", False):
            cos, sin = self.rope_emb(q, T, offset=offset)
            d = q.shape[-1]
            def rotate_half(tensor):
                return torch.cat((-tensor[..., d//2:], tensor[..., :d//2]), dim=-1)
            q = q * cos[None, :, None, :] + rotate_half(q) * sin[None, :, None, :]
            k = k * cos[None, :, None, :] + rotate_half(k) * sin[None, :, None, :]
            
        v = self.W_v(x).view(B, T, self.H, self.d_v)
        
        # Variable-Delta-t ODE Decay
        if getattr(self.config, "use_variable_delta_t", False):
            delta_t = F.softplus(self.W_delta(x)) + 1e-3
            lambdas = torch.exp(self.W_lambda)
            alpha = torch.exp(-delta_t * lambdas[None, None, :])
        else:
            alpha = torch.sigmoid(self.W_alpha(x))
        alpha = alpha.clamp(min=1e-6, max=1.0)
            
        # [STEP-10] Boundary Clock: selectively reset fast heads (h < H//2) at boundaries
        if boundary_mask is not None:
            H_half = self.H // 2
            fast_head_mask = torch.arange(self.H, device=x.device)[None, None, :] < H_half
            reset_mask = boundary_mask[:, :, None] & fast_head_mask
            alpha = torch.where(reset_mask, torch.full_like(alpha, 0.01), alpha)
            
        # [STEP-07] Sparse Write Gate: shifted ReLU — only strong signals write to state
        _beta_raw = torch.sigmoid(self.W_beta(x))
        beta      = F.relu(_beta_raw - 0.1) / 0.9
        # [STEP-08] Per-head write scale: each head has independent amplitude control
        beta      = beta * torch.exp(self.log_beta_scale)[None, None, :]
        # [STEP-09] Phase Oscillator: periodic gating on beta
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        T_cycle = torch.exp(self.log_T_cycle)
        clock_gate = 0.5 + 0.5 * torch.cos(2.0 * math.pi * pos[:, None] / T_cycle[None, :])
        beta      = beta * clock_gate[None, :, :].to(dtype=x.dtype)
        out_gate = torch.sigmoid(self.W_out_gate(x)).view(B, T, self.H, self.d_v)

        if state is not None:
            if isinstance(state, tuple):
                S_prev, n_prev = state
            else:
                S_prev, n_prev = state, None
        else:
            S_prev, n_prev = None, None

        use_epistemic_gate = getattr(self.config, "use_epistemic_gate", True)
        has_fast_kernel = (fused_nitro_scan_with_n is not None) if use_epistemic_gate else (fused_nitro_scan is not None)

        if not self.force_sequential and has_fast_kernel and q.is_cuda:
            # FAST PATH: Triton kernel handles the expensive S recurrence
            n_grad_mode = getattr(self.config, "n_grad_mode", "detached")
            
            if n_grad_mode not in ("detached", "exact"):
                raise ValueError(f"Unsupported n_grad_mode={n_grad_mode!r}; expected 'detached' or 'exact'")

            needs_grad = torch.is_grad_enabled() and (
                q.requires_grad or k.requires_grad or v.requires_grad or alpha.requires_grad or beta.requires_grad
            )

            if use_epistemic_gate:
                out, S, n_stack_full = fused_nitro_scan_with_n(q, k, v, alpha, beta, state=S_prev, initial_n=n_prev, chunk_size=32)
                n_stack = n_stack_full.transpose(1, 2)  # [B, H, T, dk] -> [B, T, H, dk]
                if needs_grad and n_grad_mode == "detached":
                    n_stack = n_stack.detach()
                n = n_stack[:, -1, :, :]
                
                # [STEP-05] Normalize and Epistemic Gate
                n_inf  = n_stack.abs().max(dim=-1)[0]                # [B, T, H]
                denom  = torch.clamp(n_inf, min=1.0).unsqueeze(-1)   # [B, T, H, 1]
                conf   = torch.tanh(n_stack.norm(dim=-1)).unsqueeze(-1)  # [B, T, H, 1]
                out    = (out / denom) * out_gate * conf
            else:
                out, S = fused_nitro_scan(q, k, v, alpha, beta, state=S_prev, chunk_size=32)
                n = None
                out    = out * out_gate
                
            out    = out.reshape(B, T, -1)
            return self.W_o(out), (S, n)

        else:
            # SEQUENTIAL PATH: Pure PyTorch fallback
            use_epistemic_gate = getattr(self.config, "use_epistemic_gate", True)
            S = S_prev if S_prev is not None else torch.zeros(B, self.H, self.d_k, self.d_v, device=x.device, dtype=x.dtype)
            if use_epistemic_gate:
                n = n_prev if n_prev is not None else torch.zeros(B, self.H, self.d_k, device=x.device, dtype=x.dtype)
            else:
                n = None
            outputs = []
            for t in range(T):
                delta = torch.einsum('bhk,bhd->bhkd', k[:, t], v[:, t])
                S = alpha[:, t, :, None, None] * S + beta[:, t, :, None, None] * delta
                
                out_t  = torch.einsum('bhkd,bhk->bhd', S, q[:, t])
                if use_epistemic_gate:
                    n = alpha[:, t, :, None] * n + beta[:, t, :, None] * k[:, t]
                    n_inf  = n.abs().max(dim=-1)[0]                    # [B, H]
                    denom  = torch.clamp(n_inf, min=1.0).unsqueeze(-1) # [B, H, 1]
                    conf   = torch.tanh(n.norm(dim=-1)).unsqueeze(-1)   # [B, H, 1]
                    out_t  = (out_t / denom) * conf
                
                out_t = out_t * out_gate[:, t]
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
        
    def forward(self, x, state=None, boundary_mask=None, offset=0):
        m_out, new_mixer_state = self.mixer(self.norm1(x), state=state, boundary_mask=boundary_mask, offset=offset)
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

    def fuse(self, current_ns, prev_layer_ns, layer_idx):
        if prev_layer_ns is None or current_ns is None:
            return current_ns
        gate = torch.sigmoid(self.fusion_gate[layer_idx])  # [H]
        gate = gate[None, :, None, None]                   # broadcast to [1, H, 1, 1]
        S_current, n_current = current_ns
        S_prev, n_prev = prev_layer_ns
        if S_prev is None or S_current is None:
            return current_ns
        S_fused = S_current + gate * S_prev                # additive state highway
        if n_current is not None and n_prev is not None:
            n_fused = n_current + gate.squeeze(-1) * n_prev    # gate is [1, H, 1, 1], squeeze(-1) makes it [1, H, 1] for broadcasting to d_k
        else:
            n_fused = None
        return (S_fused, n_fused)

class HGDMUltimate(nn.Module):
    def __init__(self, config: HGDMConfig, force_sequential=False):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Bug 2 Fix: Configure positional embedding size from configuration to save VRAM
        if getattr(config, "use_rope", False):
            assert config.d_k % 2 == 0, "RoPE requires even d_k"
            self.pos_embedding = None
        else:
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
        
        # [STEP-10] Boundary Clock: configurable reset tokens.
        boundary_mask = torch.zeros_like(byte_seq, dtype=torch.bool)
        for token_id in getattr(self.config, "boundary_token_ids", (46, 63, 33, 10)):
            boundary_mask = boundary_mask | (byte_seq == token_id)
        
        # Bug 2 Wrap-around check: Allow arbitrary token offsets during generation without out-of-bounds errors
        if self.pos_embedding is not None:
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
            x, ns = layer(x, states[i], boundary_mask=boundary_mask, offset=offset)
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
        if max_new_bytes == 0:
            return prompt_bytes
        self.eval()
        generated = [prompt_bytes]
        temp = max(temp, 1e-4)
        logits, states = self.forward(prompt_bytes)
        next_logit = logits[:, -1, :] / temp
        next_probs = F.softmax(next_logit, dim=-1)
        next_byte = torch.multinomial(next_probs, num_samples=1)
        generated.append(next_byte)
        
        offset = prompt_bytes.shape[1]
        for _ in range(max_new_bytes - 1):
            logits, next_states = self.forward(next_byte, states, offset=offset)
            states = next_states
            next_logit = logits[:, -1, :] / temp
            next_probs = F.softmax(next_logit, dim=-1)
            next_byte = torch.multinomial(next_probs, num_samples=1)
            generated.append(next_byte)
            offset += 1
        return torch.cat(generated, dim=1)
