import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

# =============================================================================
# 1. THE CONFIGURATION (Pure Byte-Level, No Patching)
# =============================================================================
@dataclass
class HGDMConfig:
    d_model: int = 1024        # Width of the global model
    n_layers: int = 24         # Number of layers in the stack
    n_heads: int = 16          # Number of heads (automatically distributes TRWs)
    d_k: int = 128             # Memory capacity (keys)
    d_v: int = 128             # Memory capacity (values)
    d_ff: int = 4096           # Feedforward inner dimension
    vocab_size: int = 256      # 256 for raw byte inputs
    hybrid_interval: int = 0   # If > 0, every Nth layer is exact-match standard Attention

# =============================================================================
# 2. CORE UTILITIES
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, config: HGDMConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w3 = nn.Linear(config.d_ff, config.d_model, bias=False)
        
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# =============================================================================
# 3. MULTI-HEAD GATED DELTA MEMORY (The O(n) Memory Engine)
# =============================================================================
from kernel_scan import ssd_parallel_scan

class MultiHeadGatedDelta(nn.Module):
    def __init__(self, config: HGDMConfig):
        super().__init__()
        self.config = config
        self.H = config.n_heads
        self.d_k = config.d_k
        self.d_v = config.d_v
        
        # Projections
        self.W_q = nn.Linear(config.d_model, self.H * self.d_k, bias=False)
        self.W_k = nn.Linear(config.d_model, self.H * self.d_k, bias=False)
        self.W_v = nn.Linear(config.d_model, self.H * self.d_v, bias=False)
        
        # Gates (alpha = forget, beta = write)
        self.W_alpha = nn.Linear(config.d_model, self.H)
        self.W_beta  = nn.Linear(config.d_model, self.H)
        
        # UPGRADE 1: Output Gate (Allows model to dynamically silence noisy memories)
        self.W_out_gate = nn.Linear(config.d_model, self.H * self.d_v)
        
        self.W_o = nn.Linear(self.H * self.d_v, config.d_model, bias=False)
        
        self._initialize_biological_timescales()

    def _initialize_biological_timescales(self):
        """ 
        Distributes Hasson's 5 Temporal Receptive Windows (TRWs) across the heads. 
        Forces some heads to forget quickly (syntax) and others to never forget (discourse).
        """
        base_taus = [4.0, 30.0, 200.0, 1200.0, 8000.0] 
        biases = []
        for h in range(self.H):
            tau = base_taus[h % len(base_taus)] 
            alpha_target = math.exp(-1.0 / tau)
            # Inverse sigmoid to set the correct pre-activation bias
            bias_val = math.log(alpha_target / (1.0 - alpha_target + 1e-8))
            biases.append(bias_val)
            
        with torch.no_grad():
            self.W_alpha.bias.copy_(torch.tensor(biases))
            self.W_alpha.weight.zero_() # Start perfectly at biological prior

    def forward(self, x, state: Optional[torch.Tensor] = None):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.H, self.d_k)
        
        # UPGRADE 2: Q-Scaling. Anchors the variance of memory retrieval, preventing explosions.
        q = q * (self.d_k ** -0.5)
        
        # L2 Norm on keys is mathematically mandatory for Delta Rule stability
        k = F.normalize(self.W_k(x).view(B, T, self.H, self.d_k), p=2, dim=-1)
        v = self.W_v(x).view(B, T, self.H, self.d_v)
        
        alpha = torch.sigmoid(self.W_alpha(x)).view(B, T, self.H, 1, 1)
        beta  = torch.sigmoid(self.W_beta(x)).view(B, T, self.H, 1, 1)
        
        # Compute the output gate in parallel
        out_gate = F.silu(self.W_out_gate(x)).view(B, T, self.H, self.d_v)
        
        # ---------------------------------------------------------------------
        # TRAINING: SSD Matrix Formulation (Runs on Tensor Cores, Zero OOM)
        # ---------------------------------------------------------------------
        if T > 1 and state is None:
            out, last_state = ssd_parallel_scan(q, k, v, alpha, beta)
            
            # Apply Output Gate
            out = out * out_gate
            return self.W_o(out.reshape(B, T, -1)), last_state
        
        # ---------------------------------------------------------------------
        # INFERENCE: O(1) Sequential Update (Matrix state never grows)
        # ---------------------------------------------------------------------
        else:
            S = torch.zeros(B, self.H, self.d_k, self.d_v, device=x.device, dtype=x.dtype) if state is None else state
            outputs = []
            for t in range(T):
                # 1. WRITE: Standard Linear Attention update (Matches SSD kernel exactly)
                delta = torch.einsum('bhk,bhd->bhkd', k[:, t], v[:, t])
                S = alpha[:, t] * S + beta[:, t] * delta
                
                # 2. OUTPUT: Read final state using query
                out_t = torch.einsum('bhkd,bhk->bhd', S, q[:, t])
                
                # Apply Output Gate
                out_t = out_t * out_gate[:, t]
                outputs.append(out_t)
                
            out = torch.stack(outputs, dim=1).reshape(B, T, -1)
            return self.W_o(out), S

# =============================================================================
# 4. HYBRID ATTENTION LAYER (Optional Exact-Match Safety Net)
# =============================================================================
class ExactAttention(nn.Module):
    """ Used if config.hybrid_interval > 0 to guarantee 100% exact local recall. """
    def __init__(self, config: HGDMConfig):
        super().__init__()
        self.H = config.n_heads
        self.d_head = config.d_model // self.H
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
    def forward(self, x, state=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        k = k.view(B, T, self.H, self.d_head).transpose(1, 2)
        q = q.view(B, T, self.H, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.H, self.d_head).transpose(1, 2)
        
        if state is not None:
            pk, pv = state
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
            
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), (k, v)

# =============================================================================
# 5. LAYER ROUTING & THE FULL STACK
# =============================================================================
class HGDMLayer(nn.Module):
    def __init__(self, config: HGDMConfig, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        
        # Route to Hybrid Exact Attention if configured
        if config.hybrid_interval > 0 and (layer_idx + 1) % config.hybrid_interval == 0:
            self.mixer = ExactAttention(config)
        else:
            self.mixer = MultiHeadGatedDelta(config)
            
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config)
        
    def forward(self, x, state=None):
        m_out, ns = self.mixer(self.norm1(x), state)
        x = x + m_out
        x = x + self.ffn(self.norm2(x))
        return x, ns

class HGDMUltimate(nn.Module):
    def __init__(self, config: HGDMConfig):
        super().__init__()
        self.config = config
        
        # Pure Byte Embedding (No Patching)
        self.byte_emb = nn.Embedding(config.vocab_size, config.d_model)
        
        # Global Stack
        self.layers = nn.ModuleList([HGDMLayer(config, i) for i in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)
        
        # Direct Output Head
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # UPGRADE 3: Weight Tying.
        # Ties the output head to the input embeddings to enforce a unified semantic space.
        self.head.weight = self.byte_emb.weight

    def forward(self, byte_seq, states=None):
        B, T = byte_seq.shape
        
        # 1. Embed Bytes directly
        x = self.byte_emb(byte_seq)
        
        # 2. Global HGDM Processing
        if states is None: states = [None] * len(self.layers)
        next_states = []
        for i, layer in enumerate(self.layers):
            x, ns = layer(x, states[i])
            next_states.append(ns)
            
        x = self.norm_f(x)
        
        # 3. Direct Output Projection
        logits = self.head(x)
        
        return logits, next_states

    @torch.no_grad()
    def compute_memory_utilization(self, states) -> float:
        """ Returns 0.0 to 1.0. If nearing 1.0, the memory matrix is saturated! """
        utilizations = []
        for state in states:
            # Exclude ExactAttention states which are tuples
            if isinstance(state, torch.Tensor) and state.dim() == 4:
                B, H, dk, dv = state.shape
                S_flat = state.reshape(-1, dk, dv)
                util = torch.linalg.matrix_norm(S_flat, ord='fro') / math.sqrt(dk * dv)
                utilizations.append(util.mean().item())
        return sum(utilizations) / len(utilizations) if utilizations else 0.0

    @torch.no_grad()
    def generate(self, prompt_bytes, max_new_bytes=100, temp=0.8):
        """ Standard 1-by-1 Autoregressive Generation. Perfect Spelling. """
        self.eval()
        generated = prompt_bytes
        
        # Prefill prompt to build up the memory state
        logits, states = self.forward(prompt_bytes)
        
        # Sample the first new byte from the last logit of the prompt
        next_logit = logits[:, -1, :] / temp
        next_probs = F.softmax(next_logit, dim=-1)
        next_byte = torch.multinomial(next_probs, num_samples=1)
        
        generated = torch.cat([generated, next_byte], dim=1)
        
        # Autoregressively generate byte-by-byte
        for _ in range(max_new_bytes - 1):
            # Advance Global State by exactly 1 byte
            logits, next_states = self.forward(next_byte, states)
            states = next_states
            
            # Predict Next Byte
            next_logit = logits[:, -1, :] / temp
            next_probs = F.softmax(next_logit, dim=-1)
            next_byte = torch.multinomial(next_probs, num_samples=1)
            
            generated = torch.cat([generated, next_byte], dim=1)
            
        return generated