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
    use_variable_delta_t: bool = False  # Feature 2: Variable-Delta-t Decay
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
        
        # Advanced Feature 2: Variable-Delta-t Continuous Decay
        if getattr(config, "use_variable_delta_t", False):
            base_taus = [4.0, 30.0, 200.0, 1200.0, 8000.0]
            initial_lambdas = []
            for h in range(self.H):
                tau = base_taus[h % len(base_taus)]
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
        
        # Advanced Feature 2: Variable-Delta-t ODE Decay
        if getattr(self.config, "use_variable_delta_t", False):
            delta_t = F.softplus(self.W_delta(x)) + 1e-3
            lambdas = torch.exp(self.W_lambda)
            alpha = torch.exp(-delta_t * lambdas[None, None, :])
        else:
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