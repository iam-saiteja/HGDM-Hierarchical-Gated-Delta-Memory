"""
Geometric Reservoir Delta (GRD) — A New Paradigm for Sequence Modeling.
v2: Now uses the fused_nitro_scan Triton kernel for all three reservoirs.
    The Python sequential loop is completely eliminated on CUDA.

Architecture:
  Reservoir A: Near-unit-magnitude decay (long memory, no decay)
               → fused_nitro_scan with alpha = |gamma| ≈ 1
  Reservoir B: NCM Novelty-Gated Semantic
               → Two-pass: fused_nitro_scan_with_n for n_stack (novelty),
                 then fused_nitro_scan with novelty-scaled beta
  Reservoir C: CADP Contradiction-Aware Correction
               → fused_nitro_scan with contradiction-scored beta

Author: iam-saiteja / HTSPC
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

# ── Triton kernel (fast path) ─────────────────────────────────────────────────
try:
    from kernel_nitro import fused_nitro_scan, fused_nitro_scan_with_n
    _HAVE_KERNEL = True
except (ImportError, Exception):
    fused_nitro_scan = None
    fused_nitro_scan_with_n = None
    _HAVE_KERNEL = False

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
@dataclass
class GRDConfig:
    d_model:   int = 512
    n_layers:  int = 8
    n_heads:   int = 8
    d_k:       int = 64
    d_v:       int = 64
    d_ff:      int = 2048
    vocab_size: int = 256
    # Reservoir A: near-unit magnitude (long-range memory)
    osc_mag_init: float = -8.0     # log(-log(|gamma|)) → |gamma| ≈ 1
    # Reservoir B: novelty-gated semantic
    novelty_threshold: float = 0.05
    slow_tau_base: float = 200.0
    # Reservoir C: contradiction-aware correction
    correction_tau: float = 10.0
    max_position_embeddings: int = 2048

# =============================================================================
# 2. PRIMITIVES
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# =============================================================================
# 3. THE CORE: GEOMETRIC RESERVOIR MIXER (Kernel-Accelerated)
# =============================================================================
class GeometricReservoirMixer(nn.Module):
    """
    Three coupled reservoirs using the fused_nitro_scan Triton kernel.

    State: (S_A, S_B, n_B, S_C)
    ─────────────────────────────────────────────
    S_A  [B, H, d_k, d_v]  Long-memory (near-unit decay)
    S_B  [B, H, d_k, d_v]  Novelty-gated semantic
    n_B  [B, H, d_k]       Key accumulator (tracks what was written to B)
    S_C  [B, H, d_k, d_v]  Contradiction/correction

    Kernel call pattern per forward:
    ─────────────────────────────────────────────
    A: fused_nitro_scan(q, k, v, alpha_A, beta_A, state=S_A)
    B: fused_nitro_scan_with_n(q, k, v, alpha_B, beta_B, state=S_B, initial_n=n_B)
         → extract n_stack, compute novelty_t from n_{t-1}, recompute effective beta_B
    B2: fused_nitro_scan_with_n(q, k, v, alpha_B, effective_beta_B, state=S_B, initial_n=n_B)
         → final S_B output + updated n_B
    C:  fused_nitro_scan(q, k, v, alpha_C, effective_beta_C, state=S_C)
         → contradiction-weighted correction
    Composite: softmax(W_gate(x)) weighted sum of [out_A, out_B, out_C]
    """
    def __init__(self, config: GRDConfig):
        super().__init__()
        self.H    = config.n_heads
        self.d_k  = config.d_k
        self.d_v  = config.d_v
        self.novelty_threshold = config.novelty_threshold

        # ── Shared QKV projections ─────────────────────────────────────────
        self.W_q = nn.Linear(config.d_model, self.H * self.d_k, bias=False)
        self.W_k = nn.Linear(config.d_model, self.H * self.d_k, bias=False)
        self.W_v = nn.Linear(config.d_model, self.H * self.d_v, bias=False)
        self.W_o = nn.Linear(self.H * self.d_v, config.d_model, bias=False)

        # ── Reservoir A: Near-Unit Magnitude (long memory) ────────────────
        # alpha_A = exp(-exp(log_neg_log_mag)) ≈ 1.0 — memory never fades
        self.log_neg_log_mag = nn.Parameter(torch.full((self.H,), config.osc_mag_init))
        self.W_beta_A = nn.Linear(config.d_model, self.H, bias=True)

        # ── Reservoir B: NCM Novelty-Gated Semantic ───────────────────────
        self.log_lambda_B = nn.Parameter(torch.tensor(
            [math.log(1.0 / (config.slow_tau_base * (h + 1)))
             for h in range(self.H)], dtype=torch.float32
        ))
        self.W_delta_B = nn.Linear(config.d_model, self.H, bias=True)
        self.W_beta_B  = nn.Linear(config.d_model, self.H, bias=True)

        # ── Reservoir C: CADP Correction ─────────────────────────────────
        self.log_lambda_C = nn.Parameter(
            torch.full((self.H,), math.log(1.0 / config.correction_tau))
        )
        self.W_beta_C = nn.Linear(config.d_model, self.H, bias=True)

        # ── Composite NCM-Weighted Read Gate ──────────────────────────────
        self.W_read_gate = nn.Linear(config.d_model, 3, bias=True)
        self.W_out_gate  = nn.Linear(config.d_model, self.H * self.d_v, bias=True)

        # Per-reservoir RMSNorm before combining
        self.norm_A = RMSNorm(self.d_v)
        self.norm_B = RMSNorm(self.d_v)
        self.norm_C = RMSNorm(self.d_v)

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            self.W_beta_A.weight.zero_(); self.W_beta_A.bias.fill_(-3.0)
            self.W_delta_B.weight.zero_(); self.W_delta_B.bias.fill_(0.5413)
            self.W_beta_B.weight.zero_(); self.W_beta_B.bias.fill_(-1.0)
            self.W_beta_C.weight.zero_(); self.W_beta_C.bias.fill_(-6.0)
            self.W_read_gate.weight.zero_(); self.W_read_gate.bias.zero_()
            self.W_out_gate.weight.zero_(); self.W_out_gate.bias.zero_()

    def _sequential_fallback(self, q, k, v, beta_A, alpha_B, beta_B, alpha_C, beta_C,
                              read_w, out_gate, state, B, T):
        """Pure-PyTorch fallback for CPU or when kernel unavailable."""
        if state is not None:
            S_A, S_B, n_B, S_C = state
        else:
            z    = torch.zeros(B, self.H, self.d_k, self.d_v, device=q.device, dtype=q.dtype)
            S_A  = z.clone(); S_B = z.clone(); S_C = z.clone()
            n_B  = torch.zeros(B, self.H, self.d_k, device=q.device, dtype=q.dtype)

        mag   = torch.exp(-torch.exp(self.log_neg_log_mag))
        outputs = []
        for t in range(T):
            k_t = k[:, t]; v_t = v[:, t]; q_t = q[:, t]
            delta = torch.einsum('bhk,bhd->bhkd', k_t, v_t)

            # A
            ba = beta_A[:, t].view(B, self.H, 1, 1)
            S_A = mag.view(1, self.H, 1, 1) * S_A + ba * delta
            R_A = torch.einsum('bhkd,bhk->bhd', S_A, q_t)

            # B with novelty
            n_norm  = F.normalize(n_B, dim=-1, eps=1e-6)
            cos_sim = (k_t * n_norm).sum(dim=-1).clamp(0, 1)
            novelty = (1.0 - cos_sim - self.novelty_threshold).clamp(min=0.0)
            ab = alpha_B[:, t].view(B, self.H, 1, 1)
            bb = (beta_B[:, t] * novelty).view(B, self.H, 1, 1)
            S_B = ab * S_B + bb * delta
            n_B = ab.squeeze(-1) * n_B + bb.squeeze(-1) * k_t
            R_B = torch.einsum('bhkd,bhk->bhd', S_B, q_t)

            # C
            S_B_pred = torch.einsum('bhkd,bhk->bhd', S_B, k_t)
            disagreement = (v_t - S_B_pred).norm(dim=-1) / (self.d_v**0.5 + 1e-6)
            contradiction = (cos_sim * disagreement).view(B, self.H, 1, 1)
            ac = alpha_C[:, t].view(B, self.H, 1, 1)
            bc = (beta_C[:, t].view(B, self.H, 1, 1)) * contradiction
            S_C = ac * S_C + bc * delta
            R_C = torch.einsum('bhkd,bhk->bhd', S_C, q_t)

            wA = read_w[:, t, 0].view(B, 1, 1)
            wB = read_w[:, t, 1].view(B, 1, 1)
            wC = read_w[:, t, 2].view(B, 1, 1)
            R  = wA * self.norm_A(R_A) + wB * self.norm_B(R_B) + wC * self.norm_C(R_C)
            outputs.append(R * out_gate[:, t])

        out = torch.stack(outputs, dim=1).reshape(B, T, self.H * self.d_v)
        return out, (S_A, S_B, n_B, S_C)

    def forward(self, x, state=None, **kwargs):
        B, T, _ = x.shape
        q = F.normalize(self.W_q(x).view(B, T, self.H, self.d_k), dim=-1, eps=1e-6)
        k = F.normalize(self.W_k(x).view(B, T, self.H, self.d_k), dim=-1, eps=1e-6)
        v = self.W_v(x).view(B, T, self.H, self.d_v)

        # Unpack state
        if state is not None:
            S_A, S_B, n_B, S_C = state
        else:
            S_A = S_B = S_C = None
            n_B = None

        # Pre-compute all gates (fully parallel across T)
        mag     = torch.exp(-torch.exp(self.log_neg_log_mag))        # [H]
        alpha_A = mag[None, None, :].expand(B, T, -1)                # [B, T, H] constant
        beta_A  = torch.sigmoid(self.W_beta_A(x))                    # [B, T, H]

        lam_B   = torch.exp(self.log_lambda_B)                       # [H]
        dt_B    = F.softplus(self.W_delta_B(x)) + 1e-3              # [B, T, H]
        alpha_B = torch.exp(-dt_B * lam_B[None, None, :])           # [B, T, H]
        beta_B  = torch.sigmoid(self.W_beta_B(x))                   # [B, T, H]

        lam_C   = torch.exp(self.log_lambda_C)                       # [H]
        alpha_C = torch.exp(-lam_C)[None, None, :].expand(B, T, -1) # [B, T, H] constant
        beta_C  = torch.sigmoid(self.W_beta_C(x))                   # [B, T, H]

        read_w   = F.softmax(self.W_read_gate(x), dim=-1)            # [B, T, 3]
        out_gate = torch.sigmoid(self.W_out_gate(x)).view(B, T, self.H, self.d_v)

        use_kernel = _HAVE_KERNEL and q.is_cuda

        if not use_kernel:
            out, new_state = self._sequential_fallback(
                q, k, v, beta_A, alpha_B, beta_B, alpha_C, beta_C,
                read_w, out_gate, state, B, T
            )
            return self.W_o(out), new_state

        # ── FAST PATH: Triton Kernel ──────────────────────────────────────
        # All three reservoirs computed without a Python loop.

        # ── Reservoir A: Near-Unit Decay ─────────────────────────────────
        out_A, S_A_new = fused_nitro_scan(
            q, k, v, alpha_A, beta_A, state=S_A, chunk_size=32
        )
        # out_A: [B, T, H, d_v]

        # ── Reservoir B: NCM Novelty-Gated (two-pass) ────────────────────
        # Pass 1: Get n_stack to compute per-step novelty in parallel
        _, _, n_stack_raw = fused_nitro_scan_with_n(
            q, k, v, alpha_B, beta_B,
            state=S_B, initial_n=n_B, chunk_size=32
        )
        # n_stack_raw: [B, H, T, d_k]
        n_stack = n_stack_raw.transpose(1, 2)  # [B, T, H, d_k]

        # Shift n_stack by 1 to get n_{t-1} for novelty at step t
        if n_B is not None:
            n_prev_exp = n_B.unsqueeze(1)                       # [B, 1, H, d_k]
        else:
            n_prev_exp = torch.zeros(B, 1, self.H, self.d_k,
                                     device=x.device, dtype=x.dtype)
        n_shifted = torch.cat([n_prev_exp, n_stack[:, :-1]], dim=1)  # [B, T, H, d_k]

        # Compute novelty_t = 1 - cosine_sim(k_t, n_{t-1}) [parallel]
        n_norm  = F.normalize(n_shifted, dim=-1, eps=1e-6)           # [B, T, H, d_k]
        cos_sim = (k * n_norm).sum(dim=-1).clamp(0, 1)              # [B, T, H]
        novelty = (1.0 - cos_sim - self.novelty_threshold).clamp(min=0.0)

        # Pass 2: Re-run with novelty-scaled beta (this is the real S_B)
        eff_beta_B = beta_B * novelty                                # [B, T, H]
        out_B, S_B_new, n_stack_B_raw = fused_nitro_scan_with_n(
            q, k, v, alpha_B, eff_beta_B,
            state=S_B, initial_n=n_B, chunk_size=32
        )
        # Extract final n_B state: last timestep of n_stack
        n_B_new = n_stack_B_raw[:, :, -1, :]                        # [B, H, d_k]

        # ── Reservoir C: CADP Contradiction ──────────────────────────────
        # Contradiction score: high cos_sim (seen before) + different value
        # We use out_B as a proxy for S_B @ k_t (approximate but kernel-friendly)
        # disagreement = ||v_t - out_B_t|| / sqrt(d_v)
        out_B_d = out_B.detach()                                     # stop grad for score
        disagreement = (v - out_B_d).norm(dim=-1) / (self.d_v**0.5 + 1e-6)
        contradiction = cos_sim * disagreement                       # [B, T, H]
        eff_beta_C = beta_C * contradiction
        out_C, S_C_new = fused_nitro_scan(
            q, k, v, alpha_C, eff_beta_C, state=S_C, chunk_size=32
        )

        # ── NCM-Weighted Composite Read ───────────────────────────────────
        # Normalize each reservoir output before mixing
        out_A_n = self.norm_A(out_A)    # [B, T, H, d_v]
        out_B_n = self.norm_B(out_B)
        out_C_n = self.norm_C(out_C)

        wA = read_w[:, :, 0].unsqueeze(-1).unsqueeze(-1)   # [B, T, 1, 1]
        wB = read_w[:, :, 1].unsqueeze(-1).unsqueeze(-1)
        wC = read_w[:, :, 2].unsqueeze(-1).unsqueeze(-1)

        R_combined = wA * out_A_n + wB * out_B_n + wC * out_C_n   # [B, T, H, d_v]
        out = (R_combined * out_gate).reshape(B, T, self.H * self.d_v)

        new_state = (S_A_new, S_B_new, n_B_new, S_C_new)
        return self.W_o(out), new_state


# =============================================================================
# 4. GRD LAYER
# =============================================================================
class GRDLayer(nn.Module):
    def __init__(self, config: GRDConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.mixer = GeometricReservoirMixer(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn   = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x, state=None, **kwargs):
        m_out, new_state = self.mixer(self.norm1(x), state=state)
        x = x + m_out
        x = x + self.ffn(self.norm2(x))
        return x, new_state


# =============================================================================
# 5. GRD FULL MODEL
# =============================================================================
class GRDModel(nn.Module):
    def __init__(self, config: GRDConfig):
        super().__init__()
        self.config    = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers    = nn.ModuleList([GRDLayer(config) for _ in range(config.n_layers)])
        self.norm_f    = RMSNorm(config.d_model)
        self.fc_out    = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.fc_out.weight = self.embedding.weight
        self._init_model()

    def _init_model(self):
        with torch.no_grad():
            nn.init.normal_(self.embedding.weight, std=0.02)
            for layer in self.layers:
                nn.init.normal_(layer.mixer.W_o.weight,
                                std=0.02 / math.sqrt(2 * self.config.n_layers))

    def forward(self, byte_seq, states=None, **kwargs):
        B, T = byte_seq.shape
        x = self.embedding(byte_seq)
        if states is None:
            states = [None] * len(self.layers)
        next_states = []
        for i, layer in enumerate(self.layers):
            x, ns = layer(x, state=states[i])
            next_states.append(ns)
        x = self.norm_f(x)
        return self.fc_out(x), next_states

    @torch.no_grad()
    def generate(self, prompt_bytes, max_new_bytes=200, temp=0.8):
        self.eval()
        temp = max(temp, 1e-4)
        generated = [prompt_bytes]
        logits, states = self.forward(prompt_bytes)
        next_byte = torch.multinomial(F.softmax(logits[:, -1] / temp, dim=-1), 1)
        generated.append(next_byte)
        for _ in range(max_new_bytes - 1):
            logits, states = self.forward(next_byte, states)
            next_byte = torch.multinomial(F.softmax(logits[:, -1] / temp, dim=-1), 1)
            generated.append(next_byte)
        return torch.cat(generated, dim=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    kernel_status = "WITH Triton kernel" if _HAVE_KERNEL else "WITHOUT kernel (CPU fallback)"
    print(f"GRD v2 — {kernel_status}")
    cfg   = GRDConfig(d_model=256, n_layers=4, n_heads=4, d_k=64, d_v=64, d_ff=512)
    model = GRDModel(cfg)
    print(f"Parameters: {count_parameters(model):,}")
    x = torch.randint(0, 256, (2, 128))
    logits, states = model(x)
    print(f"Logits: {logits.shape}  State len: {len(states[0])}")
    print("Sanity check PASSED.")
