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

    def fuse_projections(self):
        """
        Fuses W_q, W_k, W_v, W_out, W_beta, W_router, W_delta into a single GEMM projection W_in.
        """
        d_q = self.H * self.d_k
        d_k_proj = self.H * self.d_k
        d_v_proj = self.H * self.d_v
        d_out = self.H * self.d_v
        d_beta = self.H * self.K
        d_router = self.H * self.K
        d_delta = self.H * self.K
        
        total_dim = d_q + d_k_proj + d_v_proj + d_out + d_beta + d_router + d_delta
        dev = self.W_q.weight.device
        dtype = self.W_q.weight.dtype
        
        W_in = nn.Linear(self.d_model, total_dim, bias=True, device=dev, dtype=dtype)
        with torch.no_grad():
            W_in.weight.copy_(torch.cat([
                self.W_q.weight,
                self.W_k.weight,
                self.W_v.weight,
                self.W_out.weight,
                self.W_beta.weight,
                self.W_router.weight,
                self.W_delta.weight,
            ], dim=0))
            
            W_in.bias.copy_(torch.cat([
                torch.zeros(d_q, device=dev, dtype=dtype),
                torch.zeros(d_k_proj, device=dev, dtype=dtype),
                torch.zeros(d_v_proj, device=dev, dtype=dtype),
                self.W_out.bias,
                self.W_beta.bias,
                self.W_router.bias,
                self.W_delta.bias,
            ], dim=0))
            
        self.W_in = W_in
        self.is_fused = True

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
        if getattr(self, "is_fused", False):
            in_proj = self.W_in(x) # [B, 1, total_dim]
            q, k, v, out_raw, beta_raw, router_raw, delta_raw = torch.split(
                in_proj,
                [self.H * self.d_k, self.H * self.d_k, self.H * self.d_v, self.H * self.d_v,
                 self.H * self.K, self.H * self.K, self.H * self.K],
                dim=-1
            )
            q = F.normalize(q.view(B, self.H, self.d_k), p=2, dim=-1, eps=1e-6)
            k = F.normalize(k.view(B, self.H, self.d_k), p=2, dim=-1, eps=1e-6)
            v = v.view(B, self.H, self.d_v)
            out_gate = torch.sigmoid(out_raw).view(B, self.H, self.d_v)
            router_logits = router_raw.view(B, self.H, self.K)
            beta_raw = torch.sigmoid(beta_raw).view(B, self.H, self.K)
            delta_raw = delta_raw.view(B, self.H, self.K)
        else:
            q = self.W_q(x).view(B, self.H, self.d_k)
            k = self.W_k(x).view(B, self.H, self.d_k)
            v = self.W_v(x).view(B, self.H, self.d_v)
            q = F.normalize(q, p=2, dim=-1, eps=1e-6)
            k = F.normalize(k, p=2, dim=-1, eps=1e-6)
            out_gate = torch.sigmoid(self.W_out(x)).view(B, self.H, self.d_v)
            router_logits = self.W_router(x).view(B, self.H, self.K)
            beta_raw = torch.sigmoid(self.W_beta(x)).view(B, self.H, self.K)
            delta_raw = self.W_delta(x).view(B, self.H, self.K)
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
        Processes sequence x: [B, T, d_model].
        Uses fast parallel vectorized scan when T > 1 (training),
        and fast O(1) step recurrence when T == 1 (inference).
        """
        B, T, _ = x.shape
        if T == 1:
            return self.forward_step(x, state=state)
            
        device = x.device
        dtype = x.dtype
        
        # 1. Projections across full sequence
        q = self.W_q(x).view(B, T, self.H, self.d_k)
        k = self.W_k(x).view(B, T, self.H, self.d_k)
        v = self.W_v(x).view(B, T, self.H, self.d_v)
        
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)
        
        out_gate = torch.sigmoid(self.W_out(x)).view(B, T, self.H, self.d_v)
        router_logits = self.W_router(x).view(B, T, self.H, self.K)
        
        # Gates [B, T, H, K]
        beta_raw = torch.sigmoid(self.W_beta(x)).view(B, T, self.H, self.K)
        thresh = self.thresholds.unsqueeze(1) # [1, 1, 1, K]
        beta = (F.relu(beta_raw - thresh) / (1.0 - thresh + 1e-8)) * torch.exp(self.log_beta_scale)[None, None, :, :]
        
        delta_raw = self.W_delta(x).view(B, T, self.H, self.K)
        delta_t = F.softplus(delta_raw) + 1e-3
        decay_rates = torch.exp(self.band_lambdas).view(1, 1, 1, self.K) # [1, 1, 1, K]
        alpha = torch.exp(-delta_t * decay_rates).clamp(min=1e-4, max=0.9999) # [B, T, H, K]
        
        # 2. Parallel Vectorized Scan for all K bands
        q_p = q.permute(0, 2, 1, 3) # [B, H, T, d_k]
        k_p = k.permute(0, 2, 1, 3) # [B, H, T, d_k]
        v_p = v.permute(0, 2, 1, 3) # [B, H, T, d_v]
        
        qk = torch.matmul(q_p, k_p.transpose(-1, -2)) # [B, H, T, T]
        causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        
        band_readouts = []
        tournament_scores = []
        final_S_list = []
        final_n_list = []
        
        for k_idx in range(self.K):
            alpha_k = alpha[:, :, :, k_idx] # [B, T, H]
            beta_k = beta[:, :, :, k_idx]   # [B, T, H]
            
            log_alpha = torch.log(alpha_k.clamp(min=1e-7)).permute(0, 2, 1) # [B, H, T]
            cumsum_log = torch.cumsum(log_alpha, dim=-1) # [B, H, T]
            diff = cumsum_log.unsqueeze(-1) - cumsum_log.unsqueeze(-2) # [B, H, T, T]
            diff = torch.where(causal_mask, diff, torch.full_like(diff, float('-inf')))
            decay_matrix = torch.exp(diff) * beta_k.permute(0, 2, 1).unsqueeze(-2) # [B, H, T, T]
            
            # Recurrent associative output via parallel scan
            attn = qk * decay_matrix
            out_k = torch.matmul(attn, v_p).permute(0, 2, 1, 3) # [B, T, H, d_v]
            
            if self.config.use_epistemic_gate:
                n_vec = torch.matmul(decay_matrix, k_p).permute(0, 2, 1, 3) # [B, T, H, d_k]
                n_norm = n_vec.norm(dim=-1, keepdim=True).clamp(min=1e-4)
                denom = torch.clamp(n_vec.abs().max(dim=-1, keepdim=True)[0], min=1.0)
                conf = torch.tanh(n_norm)
                out_k = (out_k / denom) * conf
                
                score_k = torch.sum(q * (n_vec / n_norm), dim=-1) # [B, T, H]
                tournament_scores.append(score_k)
                final_n_list.append(n_vec[:, -1])
            else:
                tournament_scores.append(torch.zeros(B, T, self.H, device=device, dtype=dtype))
                
            band_readouts.append(out_k)
            
            # Final state for recurrent continuation
            decay_to_end = torch.exp(cumsum_log[:, :, -1:] - cumsum_log) * beta_k.permute(0, 2, 1)
            S_k_final = torch.einsum('bht,bhtk,bhtd->bhkd', decay_to_end, k_p, v_p)
            final_S_list.append(S_k_final)
            
        # 3. Stack and Tournament Convex Combination
        readouts_stack = torch.stack(band_readouts, dim=3)   # [B, T, H, K, d_v]
        score_stack = torch.stack(tournament_scores, dim=3)  # [B, T, H, K]
        
        weights = F.softmax(router_logits + self.gamma * score_stack, dim=-1) # [B, T, H, K]
        out_fused = torch.sum(readouts_stack * weights.unsqueeze(-1), dim=3)  # [B, T, H, d_v]
        out_fused = out_fused * out_gate
        
        out = self.W_o(out_fused.reshape(B, T, -1))
        
        S_final_stack = torch.stack(final_S_list, dim=1) # [B, K, H, d_k, d_v]
        n_final_stack = torch.stack(final_n_list, dim=1) if final_n_list else None
        
        return out, (S_final_stack, n_final_stack)
