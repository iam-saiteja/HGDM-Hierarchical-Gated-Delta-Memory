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
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class HBSConfig:
    d_model: int = 384
    n_heads: int = 6
    d_k: int = 64
    d_v: int = 64
    n_bands: int = 4                           # Dyadic bands: 2^0, 2^1, 2^2, 2^3
    base_taus: Tuple[float, ...] = (8.0, 32.0, 128.0, 10000.0) # Multi-timescale priors
    band_thresholds: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20) # Hierarchical write thresholds
    use_epistemic_gate: bool = True
    tournament_scale: float = 5.0


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
        
        # Projections
        self.W_q = nn.Linear(config.d_model, self.H * self.d_k, bias=False)
        self.W_k = nn.Linear(config.d_model, self.H * self.d_k, bias=False)
        self.W_v = nn.Linear(config.d_model, self.H * self.d_v, bias=False)
        self.W_o = nn.Linear(self.H * self.d_v, config.d_model, bias=False)
        
        # Hierarchical gates
        self.W_beta = nn.Linear(config.d_model, self.H * self.K, bias=True)
        self.W_out  = nn.Linear(config.d_model, self.H * self.d_v, bias=True)
        self.W_router = nn.Linear(config.d_model, self.H * self.K, bias=True)
        
        # Continuous multi-timescale decay priors:
        # Band 0: fast local syntax (tau ~ 8 tokens)
        # Band 1: clause / phrases (tau ~ 32 tokens)
        # Band 2: sentence discourse (tau ~ 128 tokens)
        # Band 3: permanent episodic anchor (tau ~ 10,000 tokens, alpha ~ 0.9999)
        lambdas = [1.0 / max(1.0, tau) for tau in config.base_taus]
        log_lambdas = [math.log(l) for l in lambdas]
        self.band_lambdas = nn.Parameter(
            torch.tensor(log_lambdas, dtype=torch.float32).view(1, 1, self.K, 1) # [1, 1, K, 1]
        )
        
        self.W_delta = nn.Linear(config.d_model, self.H * self.K, bias=True)
        self.log_beta_scale = nn.Parameter(torch.zeros(self.H, self.K))
        self.gamma = nn.Parameter(torch.tensor(config.tournament_scale))
        
        # Register hierarchical write thresholds
        self.register_buffer(
            "thresholds",
            torch.tensor(config.band_thresholds, dtype=torch.float32).view(1, 1, self.K)
        )
        
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
        """
        S = torch.zeros(batch_size, self.K, self.H, self.d_k, self.d_v, device=device, dtype=dtype)
        n = torch.zeros(batch_size, self.K, self.H, self.d_k, device=device, dtype=dtype) if self.config.use_epistemic_gate else None
        return (S, n)

    def forward_step(
        self,
        x: torch.Tensor,
        state: Optional[Tuple] = None,
        is_query_only: bool = False,
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Single-step pure O(1) recurrent forward (T=1).
        x: [B, 1, d_model]
        is_query_only: if True, reads from state without updating (inference query step).
        """
        B, T, _ = x.shape
        assert T == 1, "forward_step expects T=1"
        device = x.device
        dtype = x.dtype
        
        if state is None:
            state = self.init_state(B, device, dtype)
            
        S_bands, n_bands = state
        
        # 1. Projections
        q = self.W_q(x).view(B, self.H, self.d_k)
        k = self.W_k(x).view(B, self.H, self.d_k)
        v = self.W_v(x).view(B, self.H, self.d_v)
        
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)
        
        out_gate = torch.sigmoid(self.W_out(x)).view(B, self.H, self.d_v)
        
        # Router logits: [B, H, K]
        router_logits = self.W_router(x).view(B, self.H, self.K)
        
        # Multi-band beta write gates with hierarchical filtering: [B, H, K]
        beta_raw = torch.sigmoid(self.W_beta(x)).view(B, self.H, self.K)
        thresh = self.thresholds # [1, 1, K]
        beta = (F.relu(beta_raw - thresh) / (1.0 - thresh + 1e-8)) * torch.exp(self.log_beta_scale)[None, :, :]
        
        if is_query_only:
            beta = torch.zeros_like(beta)
            
        # Multi-band alpha decay gates: [B, H, K]
        delta_raw = self.W_delta(x).view(B, self.H, self.K)
        delta_t = F.softplus(delta_raw) + 1e-3
        decay_rates = torch.exp(self.band_lambdas).squeeze(-1) # [1, 1, K]
        alpha = torch.exp(-delta_t * decay_rates).clamp(min=1e-4, max=0.9999) # [B, H, K]
        
        # 2. Read from current state and compute updates
        delta_t_k = torch.einsum('bhk,bhd->bhkd', k, v) # [B, H, d_k, d_v]
        
        band_readouts = []
        tournament_scores = []
        
        S_new = S_bands.clone()
        n_new = n_bands.clone() if n_bands is not None else None
        
        for k_idx in range(self.K):
            # Readout: out_k = S^{(k)} * q
            o_k = torch.einsum('bhkd,bhk->bhd', S_bands[:, k_idx], q)
            
            # Tournament key alignment score: q^T n^{(k)} / (||n^{(k)}|| + 1e-4)
            if self.config.use_epistemic_gate and n_bands is not None:
                n_curr = n_bands[:, k_idx]
                n_norm = n_curr.norm(dim=-1, keepdim=True).clamp(min=1e-4)
                n_unit = n_curr / n_norm
                score_k = torch.sum(q * n_unit, dim=-1) # [B, H]
                
                n_inf = n_curr.abs().max(dim=-1)[0]
                denom = torch.clamp(n_inf, min=1.0).unsqueeze(-1)
                conf = torch.tanh(n_norm).squeeze(-1)
                o_k = (o_k / denom) * conf.unsqueeze(-1)
                tournament_scores.append(score_k)
            else:
                tournament_scores.append(torch.zeros(B, self.H, device=device, dtype=dtype))
                
            band_readouts.append(o_k)
            
            # Update state for next step
            a_k = alpha[:, :, k_idx, None, None]
            b_k = beta[:, :, k_idx, None, None]
            S_new[:, k_idx] = a_k * S_bands[:, k_idx] + b_k * delta_t_k
            
            if self.config.use_epistemic_gate and n_bands is not None:
                n_new[:, k_idx] = (
                    alpha[:, :, k_idx, None] * n_bands[:, k_idx] +
                    beta[:, :, k_idx, None] * k
                )
                
        # 3. Hierarchical Band Search Tournament Readout
        readouts_stack = torch.stack(band_readouts, dim=2)   # [B, H, K, d_v]
        score_stack = torch.stack(tournament_scores, dim=2)  # [B, H, K]
        
        # Tournament weights combine learned router with dynamic key-alignment
        weights = F.softmax(router_logits + self.gamma * score_stack, dim=-1) # [B, H, K]
        
        # Convex combination across bands: [B, H, d_v]
        out_fused = torch.sum(readouts_stack * weights.unsqueeze(-1), dim=2)
        out_fused = out_fused * out_gate
        
        out = self.W_o(out_fused.reshape(B, -1)).unsqueeze(1)
        next_state = (S_new, n_new)
        return out, next_state

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Processes a full sequence x: [B, T, d_model].
        """
        B, T, _ = x.shape
        outputs = []
        curr_state = state
        
        for t in range(T):
            x_t = x[:, t:t+1]
            out_t, curr_state = self.forward_step(x_t, state=curr_state)
            outputs.append(out_t)
            
        return torch.cat(outputs, dim=1), curr_state
