import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class HGDMConfig:
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_k: int = 64
    d_v: int = 64
    d_ff: int = 1536
    patch_size: int = 1 
    vocab_size: int = 256

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class MultiHeadGatedDelta(nn.Module):
    def __init__(self, config: HGDMConfig):
        super().__init__()
        self.H, self.d_k, self.d_v = config.n_heads, config.d_k, config.d_v
        self.W_q = nn.Linear(config.d_model, self.H * self.d_k, bias=False)
        self.W_k = nn.Linear(config.d_model, self.H * self.d_k, bias=False)
        self.W_v = nn.Linear(config.d_model, self.H * self.d_v, bias=False)
        self.W_alpha = nn.Linear(config.d_model, self.H)
        self.W_beta  = nn.Linear(config.d_model, self.H)
        self.W_o = nn.Linear(self.H * self.d_v, config.d_model, bias=False)

    def forward(self, x, state: Optional[torch.Tensor] = None):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.H, self.d_k)
        k = F.normalize(self.W_k(x).view(B, T, self.H, self.d_k), p=2, dim=-1)
        v = self.W_v(x).view(B, T, self.H, self.d_v)
        alpha = torch.sigmoid(self.W_alpha(x)).view(B, T, self.H, 1, 1)
        beta  = torch.sigmoid(self.W_beta(x)).view(B, T, self.H, 1, 1)
        if T > 1 and state is None:
            inputs = beta * (k.unsqueeze(-1) * v.unsqueeze(-2))
            log_alpha = torch.log(alpha.clamp(min=1e-8))
            cum_log_alpha = torch.cumsum(log_alpha, dim=1)
            alpha_exp = torch.exp(cum_log_alpha)
            S_t = torch.cumsum(inputs / (alpha_exp + 1e-8), dim=1) * alpha_exp
            out = torch.einsum('bthkd,bthk->bthd', S_t, q)
            return self.W_o(out.reshape(B, T, -1)), S_t[:, -1]
        else:
            S = torch.zeros(B, self.H, self.d_k, self.d_v, device=x.device, dtype=x.dtype) if state is None else state
            outputs = []
            for t in range(T):
                v_old = torch.einsum('bhkd,bhk->bhd', S, k[:, t])
                S = alpha[:, t] * S + beta[:, t] * torch.einsum('bhk,bhd->bhkd', k[:, t], v[:, t] - v_old)
                outputs.append(torch.einsum('bhkd,bhk->bhd', S, q[:, t]))
            return self.W_o(torch.stack(outputs, dim=1).reshape(B, T, -1)), S

class SwiGLU(nn.Module):
    def __init__(self, config: HGDMConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w3 = nn.Linear(config.d_ff, config.d_model, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class HGDMLayer(nn.Module):
    def __init__(self, config: HGDMConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model); self.mixer = MultiHeadGatedDelta(config)
        self.norm2 = RMSNorm(config.d_model); self.ffn = SwiGLU(config)
    def forward(self, x, state=None):
        m_out, ns = self.mixer(self.norm1(x), state)
        x = x + m_out
        x = x + self.ffn(self.norm2(x))
        return x, ns

class HGDMPatch(nn.Module):
    def __init__(self, config: HGDMConfig):
        super().__init__()
        self.byte_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([HGDMLayer(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model); self.head = nn.Linear(config.d_model, config.vocab_size)
        
    def forward(self, x, states=None):
        x = self.byte_emb(x)
        if states is None: states = [None] * len(self.layers)
        next_states = []
        for i, layer in enumerate(self.layers):
            x, ns = layer(x, states[i]); next_states.append(ns)
        return self.head(self.norm_f(x)), next_states

    @torch.no_grad()
    def generate(self, prompt_bytes, max_new_bytes=100, temp=0.8):
        self.eval()
        device = next(self.parameters()).device
        generated = prompt_bytes
        
        _, states = self.forward(prompt_bytes)
        
        for _ in range(max_new_bytes):
            last_byte = generated[:, -1:]
            logits, next_states = self.forward(last_byte, states)
            states = next_states
            
            logits = logits[:, -1, :] / temp
            probs = F.softmax(logits, dim=-1)
            next_byte = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_byte], dim=1)
            
        return generated
