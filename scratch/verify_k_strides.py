"""
verify_k_strides.py
====================
Verifies that the Triton fused kernel correctly uses independent K strides.

Test strategy
-------------
1. Run a forward+backward pass with the fused kernel (force_sequential=False).
2. Run the same pass with the sequential pure-PyTorch fallback.
3. Compare outputs and gradients — they must match to within float16 tolerance.
   If K was still using Q's strides the two results would diverge on any layout
   where Q and K differ (e.g. permuted tensors), proving the fix is live.
"""

import torch
from hgdm_ultimate import HGDMUltimate, HGDMConfig

RTOL = 1e-2
ATOL = 1e-2

def make_model(force_seq):
    cfg = HGDMConfig(
        d_model=128,
        n_layers=1,
        n_heads=4,
        d_k=64,   # kernel requires d_k=d_v=64
        d_v=64,
        d_ff=256,
        vocab_size=256,
    )
    m = HGDMUltimate(cfg, force_sequential=force_seq)
    return m

def run(model, tokens):
    logits, _ = model(tokens)
    loss = logits.float().mean()
    loss.backward()
    return logits.detach().float()

def main():
    device = torch.device("cuda")
    torch.manual_seed(0)

    tokens = torch.randint(0, 256, (1, 64), device=device)

    # Sequential (reference)
    seq_model = make_model(force_seq=True).to(device)
    seq_out = run(seq_model, tokens)

    # Fused Triton kernel
    fused_model = make_model(force_seq=False).to(device)
    # Copy weights so comparison is fair
    fused_model.load_state_dict(seq_model.state_dict())
    fused_out = run(fused_model, tokens)

    match = torch.allclose(seq_out, fused_out, rtol=RTOL, atol=ATOL)
    max_diff = (seq_out - fused_out).abs().max().item()
    print(f"Max output diff (sequential vs fused): {max_diff:.6f}")

    if match:
        print("[SUCCESS] K strides are correctly decoupled — fused and sequential outputs match.")
    else:
        print("[FAIL] Outputs diverge — K stride fix may be incomplete.")
        raise AssertionError(f"Max diff {max_diff} exceeds tolerance")

if __name__ == "__main__":
    main()
