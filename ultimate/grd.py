"""
Geometric Reservoir Delta (GRD) — A New Paradigm for Sequence Modeling.

Replaces Transformer Self-Attention with three geometrically coupled reservoirs,
fusing Native Cognitive Memory (NCM) geometry directly into the ODE state.

Architecture:
  Reservoir A: Complex Oscillatory  — syntax, never decays (rotates on unit circle)
  Reservoir B: NCM Novelty-Gated   — semantics, writes only novel information
  Reservoir C: CADP Correction     — detects contradictions, holds corrections

Author: iam-saiteja / HTSPC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

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
    osc_mag_init: float = -8.0
    novelty_threshold: float = 0.05
    slow_tau_base: float = 200.0
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
# 3. THE CORE: GEOMETRIC RESERVOIR MIXER
# =============================================================================
class GeometricReservoirMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.H    = config.n_heads
        self.d_k  = config.d_k
        self.d_v  = config.d_v
        self.novelty_threshold = config.novelty_threshold

        self.W_q = nn.Linear(config.d_model, self.H * self.d_k, bias=False)
        self.W_k = nn.Linear(config.d_model, self.H * self.d_k, bias=False)
        self.W_v = nn.Linear(config.d_model, self.H * self.d_v, bias=False)
        self.W_o = nn.Linear(self.H * self.d_v, config.d_model, bias=False)

        # Reservoir A: Complex Oscillator
        self.log_neg_log_mag = nn.Parameter(torch.full((self.H,), config.osc_mag_init))
        freqs = torch.linspace(0.02 * math.pi, 0.98 * math.pi, self.H)
        self.log_freq = nn.Parameter(torch.log(freqs.clamp(min=1e-4)))
        self.W_beta_A = nn.Linear(config.d_model, self.H, bias=True)

        # Reservoir B: NCM Novelty-Gated Semantic
        self.log_lambda_B = nn.Parameter(torch.tensor(
            [math.log(1.0 / (config.slow_tau_base * (h + 1))) for h in range(self.H)],
            dtype=torch.float32
        ))
        self.W_delta_B = nn.Linear(config.d_model, self.H, bias=True)
        self.W_beta_B  = nn.Linear(config.d_model, self.H, bias=True)

        # Reservoir C: CADP Correction
        self.log_lambda_C = nn.Parameter(
            torch.full((self.H,), math.log(1.0 / config.correction_tau))
        )
        self.W_beta_C = nn.Linear(config.d_model, self.H, bias=True)

        # Composite Read Gate
        self.W_read_gate = nn.Linear(config.d_model, 3, bias=True)
        self.W_out_gate  = nn.Linear(config.d_model, self.H * self.d_v, bias=True)

        # Per-reservoir normalization before composite mix
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

    def forward(self, x, state=None, **kwargs):
        B, T, _ = x.shape
        q = F.normalize(self.W_q(x).view(B, T, self.H, self.d_k), dim=-1, eps=1e-6)
        k = F.normalize(self.W_k(x).view(B, T, self.H, self.d_k), dim=-1, eps=1e-6)
        v = self.W_v(x).view(B, T, self.H, self.d_v)

        if state is not None:
            S_A_real, S_A_imag, S_B, n_B, S_C = state
        else:
            z    = torch.zeros(B, self.H, self.d_k, self.d_v, device=x.device, dtype=x.dtype)
            S_A_real = z.clone(); S_A_imag = z.clone()
            S_B = z.clone(); S_C = z.clone()
            n_B = torch.zeros(B, self.H, self.d_k, device=x.device, dtype=x.dtype)

        # Pre-compute per-sequence gates (vectorized across T)
        mag    = torch.exp(-torch.exp(self.log_neg_log_mag))  # [H]
        freq   = torch.exp(self.log_freq)                     # [H]
        r_cos  = (mag * torch.cos(freq))                      # [H]
        r_sin  = (mag * torch.sin(freq))                      # [H]
        beta_A = torch.sigmoid(self.W_beta_A(x))             # [B, T, H]

        lam_B   = torch.exp(self.log_lambda_B)               # [H]
        dt_B    = F.softplus(self.W_delta_B(x)) + 1e-3       # [B, T, H]
        alpha_B = torch.exp(-dt_B * lam_B[None, None, :])    # [B, T, H]
        beta_B  = torch.sigmoid(self.W_beta_B(x))            # [B, T, H]

        lam_C   = torch.exp(self.log_lambda_C)               # [H]
        alpha_C = torch.exp(-lam_C)                          # [H] fixed
        beta_C  = torch.sigmoid(self.W_beta_C(x))            # [B, T, H]

        read_w   = F.softmax(self.W_read_gate(x), dim=-1)    # [B, T, 3]
        out_gate = torch.sigmoid(self.W_out_gate(x)).view(B, T, self.H, self.d_v)

        outputs = []
        for t in range(T):
            k_t = k[:, t]; v_t = v[:, t]; q_t = q[:, t]
            delta = torch.einsum('bhk,bhd->bhkd', k_t, v_t)  # [B, H, d_k, d_v]

            # ── Reservoir A: Complex Rotation ────────────────────────────
            rc = r_cos.view(1, self.H, 1, 1)
            rs = r_sin.view(1, self.H, 1, 1)
            ba = beta_A[:, t].view(B, self.H, 1, 1)

            new_A_real = rc * S_A_real - rs * S_A_imag + ba * delta
            new_A_imag = rs * S_A_real + rc * S_A_imag
            S_A_real   = new_A_real
            S_A_imag   = new_A_imag
            R_A = torch.einsum('bhkd,bhk->bhd', S_A_real, q_t)

            # ── Reservoir B: NCM Novelty-Gated ───────────────────────────
            n_norm  = F.normalize(n_B, dim=-1, eps=1e-6)
            cos_sim = (k_t * n_norm).sum(dim=-1).clamp(0, 1)  # [B, H]
            novelty = (1.0 - cos_sim - self.novelty_threshold).clamp(min=0.0)
            novelty = novelty.view(B, self.H, 1, 1)

            ab = alpha_B[:, t].view(B, self.H, 1, 1)
            bb = beta_B[:, t].view(B, self.H, 1, 1) * novelty

            S_B = ab * S_B + bb * delta
            n_B = ab.squeeze(-1) * n_B + bb.squeeze(-1) * k_t
            R_B = torch.einsum('bhkd,bhk->bhd', S_B, q_t)

            # ── Reservoir C: CADP Contradiction ──────────────────────────
            S_B_pred     = torch.einsum('bhkd,bhk->bhd', S_B, k_t)
            disagreement = (v_t - S_B_pred).norm(dim=-1) / (self.d_v ** 0.5 + 1e-6)
            contradiction_score = (cos_sim * disagreement).view(B, self.H, 1, 1)

            ac = alpha_C.view(1, self.H, 1, 1)
            bc = beta_C[:, t].view(B, self.H, 1, 1) * contradiction_score

            S_C = ac * S_C + bc * delta
            R_C = torch.einsum('bhkd,bhk->bhd', S_C, q_t)

            # ── NCM-Weighted Composite Read ───────────────────────────────
            R_A_n = self.norm_A(R_A)
            R_B_n = self.norm_B(R_B)
            R_C_n = self.norm_C(R_C)

            wA = read_w[:, t, 0].view(B, 1, 1)
            wB = read_w[:, t, 1].view(B, 1, 1)
            wC = read_w[:, t, 2].view(B, 1, 1)

            R_combined = wA * R_A_n + wB * R_B_n + wC * R_C_n
            out_t = R_combined * out_gate[:, t]
            outputs.append(out_t)

        out = torch.stack(outputs, dim=1).reshape(B, T, self.H * self.d_v)
        return self.W_o(out), (S_A_real, S_A_imag, S_B, n_B, S_C)


# =============================================================================
# 4. GRD LAYER
# =============================================================================
class GRDLayer(nn.Module):
    def __init__(self, config):
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
    """
    Full Geometric Reservoir Delta language model (byte-level, vocab_size=256).
    No positional embeddings needed — Reservoir A provides temporal encoding
    through complex oscillations across heads at different frequencies.
    """
    def __init__(self, config):
        super().__init__()
        self.config    = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers    = nn.ModuleList([GRDLayer(config) for _ in range(config.n_layers)])
        self.norm_f    = RMSNorm(config.d_model)
        self.fc_out    = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.fc_out.weight = self.embedding.weight  # weight tying
        self._init_model()

    def _init_model(self):
        with torch.no_grad():
            nn.init.normal_(self.embedding.weight, std=0.02)
            for layer in self.layers:
                nn.init.normal_(layer.mixer.W_o.weight,
                                std=0.02 / math.sqrt(2 * self.config.n_layers))

    def forward(self, byte_seq, states=None, **kwargs):
        B, T = byte_seq.shape
        x    = self.embedding(byte_seq)
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
    total = sum(p.numel() for p in model.parameters())
    return total


if __name__ == "__main__":
    cfg   = GRDConfig(d_model=256, n_layers=4, n_heads=4, d_k=64, d_v=64, d_ff=512)
    model = GRDModel(cfg)
    print(f"GRDModel | Params: {count_parameters(model):,}")
    x = torch.randint(0, 256, (2, 128))
    logits, states = model(x)
    print(f"Logits: {logits.shape}  State A_real: {states[0][0].shape}")
    prompt = torch.randint(0, 256, (1, 10))
    out = model.generate(prompt, max_new_bytes=20)
    print(f"Generated {out.shape[1]} bytes. PASSED.")
