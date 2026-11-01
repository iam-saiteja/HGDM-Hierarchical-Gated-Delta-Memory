"""
Hierarchical Band Search (HBS) Memory Engine.
Implements a multi-scale dyadic binary state tree (strides 2^0, 2^1, 2^2, 2^3)
to eliminate linear superposition collapse and catastrophic needle-forgetting
in pure recurrent models while maintaining strict O(1) inference memory and zero attention matrices.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class HBSConfig:
    d_model: int = 384
    n_heads: int = 6
    d_k: int = 64
    d_v: int = 64
    n_bands: int = 4              # Dyadic bands: 2^0, 2^1, 2^2, 2^3 -> [1, 2, 4, 8]
    strides: Tuple[int, ...] = (1, 2, 4, 8)
    base_taus: Tuple[float, ...] = (8.0, 32.0, 128.0, 10000.0) # Timescales per band
    use_epistemic_gate: bool = True
    sparse_write_threshold: float = 0.15


class HierarchicalBandMemory(nn.Module):
    """
    Hierarchical Band Search (HBS) Recurrent Mixer.
    Decomposes flat state into K dyadic temporal bands with logarithmic hop retrieval.
    """
    def __init__(self, config: HBSConfig):
        super().__init__()
        self.config = config
        self.H = config.n_heads
        self.d_k = config.d_k
        self.d_v = config.d_v
        self.d_model = config.d_model
        self.K = config.n_bands
        self.strides = config.strides
        
        # Projections
        self.W_q = nn.Linear(config.d_model, self.H * self.d_k, bias=False)
        self.W_k = nn.Linear(config.d_model, self.H * self.d_k, bias=False)
        self.W_v = nn.Linear(config.d_model, self.H * self.d_v, bias=False)
        self.W_o = nn.Linear(self.H * self.d_v, config.d_model, bias=False)
        
        # Gates
        self.W_beta = nn.Linear(config.d_model, self.H * self.K, bias=True)
        self.W_out  = nn.Linear(config.d_model, self.H * self.d_v, bias=True)
        self.W_router = nn.Linear(config.d_model, self.H * self.K, bias=True)
        
        # Learnable multi-timescale decay prior for each band
        # Band 0: fast (tau ~ 8)
        # Band 1: medium (tau ~ 32)
        # Band 2: slow (tau ~ 128)
        # Band 3: permanent anchor (tau ~ 10,000, zero decay)
        lambdas = [1.0 / max(1.0, tau) for tau in config.base_taus]
        self.band_lambdas = nn.Parameter(
            torch.tensor(lambdas, dtype=torch.float32).view(1, 1, self.K, 1) # [1, 1, K, 1]
        )
        
        self.W_delta = nn.Linear(config.d_model, self.H * self.K, bias=True)
        self.log_beta_scale = nn.Parameter(torch.zeros(self.H, self.K))
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.W_q.weight, std=0.02)
        nn.init.normal_(self.W_k.weight, std=0.02)
        nn.init.normal_(self.W_v.weight, std=0.02)
        nn.init.normal_(self.W_o.weight, std=0.02)
        
        nn.init.zeros_(self.W_beta.weight)
        nn.init.constant_(self.W_beta.bias, 0.0)
        
        nn.init.zeros_(self.W_out.weight)
        nn.init.constant_(self.W_out.bias, 2.0)
        
        nn.init.zeros_(self.W_router.weight)
        nn.init.zeros_(self.W_router.bias)
        
        nn.init.zeros_(self.W_delta.weight)
        nn.init.constant_(self.W_delta.bias, 0.5413)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        """
        Allocates the O(1) multi-scale state hierarchy:
        - S_bands: [B, K, H, d_k, d_v]
        - n_bands: [B, K, H, d_k]
        - accum_k: list of [B, H, d_k] per band
        - accum_v: list of [B, H, d_v] per band
        - step_counter: int
        """
        S = torch.zeros(batch_size, self.K, self.H, self.d_k, self.d_v, device=device, dtype=dtype)
        n = torch.zeros(batch_size, self.K, self.H, self.d_k, device=device, dtype=dtype)
        accum_k = [torch.zeros(batch_size, self.H, self.d_k, device=device, dtype=dtype) for _ in range(self.K)]
        accum_v = [torch.zeros(batch_size, self.H, self.d_v, device=device, dtype=dtype) for _ in range(self.K)]
        step_counter = 0
        return (S, n, accum_k, accum_v, step_counter)

    def forward_step(
        self,
        x: torch.Tensor,
        state: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Single-step pure O(1) recurrent forward (T=1).
        x: [B, 1, d_model]
        """
        B, T, _ = x.shape
        assert T == 1, "forward_step expects T=1"
        device = x.device
        dtype = x.dtype
        
        if state is None:
            state = self.init_state(B, device, dtype)
            
        S_bands, n_bands, accum_k, accum_v, step = state
        step += 1
        
        # 1. Projections
        q = self.W_q(x).view(B, self.H, self.d_k)
        k = self.W_k(x).view(B, self.H, self.d_k)
        v = self.W_v(x).view(B, self.H, self.d_v)
        
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)
        
        out_gate = torch.sigmoid(self.W_out(x)).view(B, self.H, self.d_v)
        
        # Router logits: [B, H, K]
        router_logits = self.W_router(x).view(B, self.H, self.K)
        
        # Multi-band beta write gates: [B, H, K]
        beta_raw = torch.sigmoid(self.W_beta(x)).view(B, self.H, self.K)
        thresh = self.config.sparse_write_threshold
        beta = (F.relu(beta_raw - thresh) / (1.0 - thresh + 1e-8)) * torch.exp(self.log_beta_scale)[None, :, :]
        
        # Multi-band alpha decay gates: [B, H, K]
        delta_raw = self.W_delta(x).view(B, self.H, self.K)
        delta_t = F.softplus(delta_raw) + 1e-3
        decay_rates = torch.exp(self.band_lambdas).squeeze(-1) # [1, 1, K]
        alpha = torch.exp(-delta_t * decay_rates).clamp(min=1e-4, max=0.9999) # [B, H, K]
        
        # 2. Update each dyadic band in the hierarchy
        # Band readouts
        band_readouts = []
        band_confidences = []
        
        S_new = S_bands.clone()
        n_new = n_bands.clone()
        
        for k_idx, stride in enumerate(self.strides):
            # Accumulate token features for this band
            accum_k[k_idx] = accum_k[k_idx] + k
            accum_v[k_idx] = accum_v[k_idx] + v
            
            # Check if this dyadic stride triggers a commit
            if step % stride == 0:
                # Scaled dyadic block update
                scale = 1.0 / math.sqrt(stride)
                k_block = accum_k[k_idx] * scale
                v_block = accum_v[k_idx] * scale
                
                # Outer product delta
                delta_block = torch.einsum('bhk,bhd->bhkd', k_block, v_block)
                a_k = alpha[:, :, k_idx, None, None]
                b_k = beta[:, :, k_idx, None, None]
                
                # Recurrent matrix update
                S_new[:, k_idx] = a_k * S_bands[:, k_idx] + b_k * delta_block
                
                # Epistemic normalizer vector update
                if self.config.use_epistemic_gate:
                    n_new[:, k_idx] = (
                        alpha[:, :, k_idx, None] * n_bands[:, k_idx] +
                        beta[:, :, k_idx, None] * k_block
                    )
                
                # Reset accumulators
                accum_k[k_idx].zero_()
                accum_v[k_idx].zero_()
                
            # Readout from current band state
            # out_k = S^{(k)} * q
            o_k = torch.einsum('bhkd,bhk->bhd', S_new[:, k_idx], q)
            
            # Confidence normalizer
            if self.config.use_epistemic_gate:
                n_curr = n_new[:, k_idx]
                n_inf = n_curr.abs().max(dim=-1)[0]
                denom = torch.clamp(n_inf, min=1.0).unsqueeze(-1)
                conf = torch.tanh(n_curr.norm(dim=-1)).unsqueeze(-1)
                o_k = (o_k / denom) * conf
                band_confidences.append(conf.squeeze(-1)) # [B, H]
            else:
                band_confidences.append(torch.ones(B, self.H, device=device, dtype=dtype))
                
            band_readouts.append(o_k) # [B, H, d_v]
            
        # 3. Hierarchical Band Search Tournament Readout
        # Stack readouts: [B, H, K, d_v]
        readouts_stack = torch.stack(band_readouts, dim=2)
        conf_stack = torch.stack(band_confidences, dim=2) # [B, H, K]
        
        # Tournament weights combine router prediction and epistemic confidence
        weights = F.softmax(router_logits + 0.5 * torch.log(conf_stack.clamp(min=1e-5)), dim=-1) # [B, H, K]
        
        # Convex combination across bands: [B, H, d_v]
        out_fused = torch.sum(readouts_stack * weights.unsqueeze(-1), dim=2)
        out_fused = out_fused * out_gate
        
        # Out projection: [B, 1, d_model]
        out = self.W_o(out_fused.reshape(B, -1)).unsqueeze(1)
        next_state = (S_new, n_new, accum_k, accum_v, step)
        
        return out, next_state

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Processes a full sequence x: [B, T, d_model].
        Unrolls recurrent dyadic bands step-by-step to guarantee identical state tracking.
        """
        B, T, _ = x.shape
        outputs = []
        curr_state = state
        
        for t in range(T):
            x_t = x[:, t:t+1]
            out_t, curr_state = self.forward_step(x_t, state=curr_state)
            outputs.append(out_t)
            
        return torch.cat(outputs, dim=1), curr_state
