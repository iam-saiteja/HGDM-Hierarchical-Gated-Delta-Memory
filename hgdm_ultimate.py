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
    hybrid_interval: int = 0   

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
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

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
        self.W_alpha = nn.Linear(config.d_model, self.H)
        self.W_beta  = nn.Linear(config.d_model, self.H)
        
        self.W_out_gate = nn.Linear(config.d_model, self.H * self.d_v)
        self.W_o = nn.Linear(self.H * self.d_v, config.d_model, bias=False)
        
        self._initialize_weights()

    def _initialize_weights(self):
        base_taus = [4.0, 30.0, 200.0, 1200.0, 8000.0] 
        biases = []
        for h in range(self.H):
            tau = base_taus[h % len(base_taus)] 
            alpha_target = math.exp(-1.0 / tau)
            bias_val = math.log(alpha_target / (1.0 - alpha_target + 1e-8))
            biases.append(bias_val)
            
        with torch.no_grad():
            self.W_alpha.bias.copy_(torch.tensor(biases))
            self.W_alpha.weight.zero_() 
            self.W_q.weight.data *= 0.1
            self.W_k.weight.data *= 0.1
            self.W_beta.weight.zero_()
            self.W_beta.bias.fill_(-1.0) 
            self.W_out_gate.weight.zero_()
            self.W_out_gate.bias.zero_()

    def forward(self, x, state=None):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.H, self.d_k)
        k = self.W_k(x).view(B, T, self.H, self.d_k)
        v = self.W_v(x).view(B, T, self.H, self.d_v)
        
        alpha = torch.sigmoid(self.W_alpha(x))
        beta  = torch.sigmoid(self.W_beta(x))
        out_gate = torch.sigmoid(self.W_out_gate(x)).view(B, T, self.H, self.d_v)

        if not self.force_sequential and fused_nitro_scan is not None:
            # FAST PATH: Triton Fused Kernel
            out, S = fused_nitro_scan(q, k, v, alpha, beta, state)
            out = out * out_gate
            out = out.reshape(B, T, -1)
            return self.W_o(out), S
        else:
            # SEQUENTIAL PATH: Pure PyTorch (O(T) memory and slow, but stable fallback)
            S = state if state is not None else torch.zeros(B, self.H, self.d_k, self.d_v, device=x.device, dtype=x.dtype)
            outputs = []
            for t in range(T):
                delta = torch.einsum('bhk,bhd->bhkd', k[:, t], v[:, t])
                S = alpha[:, t, :, None, None] * S + beta[:, t, :, None, None] * delta
                out_t = torch.einsum('bhkd,bhk->bhd', S, q[:, t]) * out_gate[:, t]
                outputs.append(out_t)
            out = torch.stack(outputs, dim=1).reshape(B, T, -1)
            return self.W_o(out), S

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

class HGDMUltimate(nn.Module):
    def __init__(self, config: HGDMConfig, force_sequential=False):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 16384, config.d_model) * 0.02)
        
        self.layers = nn.ModuleList([
            HGDMLayer(config, i, force_sequential=force_sequential) for i in range(config.n_layers)
        ])
        
        self.norm_f = RMSNorm(config.d_model)
        self.fc_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.fc_out.weight = self.embedding.weight

    def forward(self, byte_seq, states=None):
        B, T = byte_seq.shape
        x = self.byte_emb(byte_seq)
        if states is None: states = [None] * len(self.layers)
        next_states = []
        for i, layer in enumerate(self.layers):
            x, ns = layer(x, states[i])
            next_states.append(ns)
        x = self.norm_f(x)
        return self.head(x), next_states

    @torch.no_grad()
    def generate(self, prompt_bytes, max_new_bytes=100, temp=0.8):
        self.eval()
        generated = prompt_bytes
        logits, states = self.forward(prompt_bytes)
        next_logit = logits[:, -1, :] / temp
        next_probs = F.softmax(next_logit, dim=-1)
        next_byte = torch.multinomial(next_probs, num_samples=1)
        generated = torch.cat([generated, next_byte], dim=1)
        for _ in range(max_new_bytes - 1):
            logits, next_states = self.forward(next_byte, states)
            states = next_states
            next_logit = logits[:, -1, :] / temp
            next_probs = F.softmax(next_logit, dim=-1)
            next_byte = torch.multinomial(next_probs, num_samples=1)
            generated = torch.cat([generated, next_byte], dim=1)
        return generated