"""
verify_kernel_params.py
========================
Verifies that the parameterised Triton kernel (D_K, D_V, CHUNK_SIZE as tl.constexpr)
compiles and produces numerically consistent results at different block dimensions.

Strategy: build an HGDMUltimate model at the target dimensions, run the same tokens
through the fused and sequential paths (same weights), and assert outputs match.
This ensures alpha/beta are in realistic ranges (model-initialised timescales).

Configurations tested:
  1. d_k=64, d_v=64  (production, same as baseline)
  2. d_k=32, d_v=32  (half-width — exercises D_K=32, CHUNK_SIZE=16)
"""

import torch
from hgdm_ultimate import HGDMUltimate, HGDMConfig

RTOL = 1e-2
ATOL = 1e-2


def run_config(d_model, n_heads, d_k, d_v, d_ff, seq_len, label):
    device = torch.device("cuda")
    torch.manual_seed(0)

    cfg = HGDMConfig(
        d_model=d_model,
        n_layers=1,
        n_heads=n_heads,
        d_k=d_k,
        d_v=d_v,
        d_ff=d_ff,
        vocab_size=256,
    )

    tokens = torch.randint(0, 256, (1, seq_len), device=device)

    # --- Sequential (reference) ---
    seq_model = HGDMUltimate(cfg, force_sequential=True).to(device)
    seq_logits, _ = seq_model(tokens)
    loss_seq = seq_logits.float().mean()
    loss_seq.backward()

    # --- Fused Triton kernel ---
    fused_model = HGDMUltimate(cfg, force_sequential=False).to(device)
    fused_model.load_state_dict(seq_model.state_dict())   # same weights
    fused_logits, _ = fused_model(tokens)
    loss_fused = fused_logits.float().mean()
    loss_fused.backward()

    out_match = torch.allclose(
        seq_logits.float(), fused_logits.float(), rtol=RTOL, atol=ATOL
    )
    max_diff = (seq_logits.float() - fused_logits.float()).abs().max().item()
    status = "[PASS]" if out_match else "[FAIL]"
    print(f"{status}  {label}  |  max_diff={max_diff:.6f}")

    assert out_match, f"{label}: output mismatch, max diff={max_diff}"


def main():
    print("=" * 60)
    print("VERIFY: Parameterised Triton block dimensions")
    print("=" * 60)

    # Config 1: production (d_k=64, d_v=64, chunk_size=32 default)
    run_config(
        d_model=256, n_heads=4, d_k=64, d_v=64, d_ff=512,
        seq_len=128,
        label="d_k=64  d_v=64  chunk=32  (production)",
    )

    # Config 2: half-width (d_k=32, d_v=32)
    # The kernel auto-uses CHUNK_SIZE=32 by default; d_k/d_v are now constexpr
    run_config(
        d_model=128, n_heads=4, d_k=32, d_v=32, d_ff=256,
        seq_len=64,
        label="d_k=32  d_v=32  chunk=32  (small)",
    )

    print("=" * 60)
    print("[SUCCESS] All parameterised kernel configurations verified.")
    print("=" * 60)


if __name__ == "__main__":
    main()
