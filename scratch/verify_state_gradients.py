"""
verify_state_gradients.py
==========================
Verifies the correctness of the cross-segment state gradient backpropagation
implemented in `kernel_nitro.py`. It compares the gradients computed by the
fused Triton kernel against a pure PyTorch sequential reference.

It uses an HGDMUltimate model to ensure all inputs (Q, K, V, Alpha, Beta)
are realistically scaled and bounded, preventing numerical instability
experienced with raw randn inputs.
"""

import torch
from hgdm_ultimate import HGDMUltimate, HGDMConfig

# Set tolerances appropriate for Triton float16/bfloat16 accumulation vs PyTorch float32
RTOL = 2e-2
ATOL = 2e-2

def run_verify():
    device = torch.device("cuda")
    torch.manual_seed(0)

    # 1. Configuration (use production sizing, small context)
    cfg = HGDMConfig(
        d_model=256,
        n_layers=1,
        n_heads=4,
        d_k=64,
        d_v=64,
        d_ff=512,
        vocab_size=256,
    )

    seq_len = 64
    tokens = torch.randint(0, 256, (1, seq_len), device=device)

    # 2. Set up initial state requiring gradients (passed as a list to the top-level model)
    state_seq   = torch.randn(1, cfg.n_heads, cfg.d_k, cfg.d_v, device=device, requires_grad=True)
    state_fused = state_seq.clone().detach().requires_grad_(True)

    # Future gradient flowing from downstream segment
    dstate_target = torch.randn(1, cfg.n_heads, cfg.d_k, cfg.d_v, device=device)

    # 3. Pure PyTorch Sequential Model Path
    seq_model = HGDMUltimate(cfg, force_sequential=True).to(device)
    out_seq, final_states_seq = seq_model(tokens, states=[state_seq])
    final_state_seq = final_states_seq[0]
    
    # Loss depends on both output and final state (generates non-None dstate)
    loss_seq = out_seq.float().sum() + (final_state_seq.float() * dstate_target).sum()
    loss_seq.backward()

    dq_seq_grad = state_seq.grad.clone()

    # 4. Fused Triton Kernel Model Path
    fused_model = HGDMUltimate(cfg, force_sequential=False).to(device)
    fused_model.load_state_dict(seq_model.state_dict())  # Copy weights exactly

    out_fused, final_states_fused = fused_model(tokens, states=[state_fused])
    final_state_fused = final_states_fused[0]
    
    loss_fused = out_fused.float().sum() + (final_state_fused.float() * dstate_target).sum()
    loss_fused.backward()

    dq_fused_grad = state_fused.grad.clone()

    # 5. Verification
    print("=" * 60)
    print("VERIFY: Cross-Segment State Gradient Backpropagation")
    print("=" * 60)

    out_diff = (out_seq.float() - out_fused.float()).abs().max().item()
    state_diff = (final_state_seq.float() - final_state_fused.float()).abs().max().item()
    grad_diff = (dq_seq_grad - dq_fused_grad).abs().max().item()

    print(f"Forward Output Max Diff:    {out_diff:.6f}")
    print(f"Forward Final State Diff:   {state_diff:.6f}")
    print(f"Gradient of Initial State:  {grad_diff:.6f}  <-- Flowing backward through segment")

    out_match = out_diff < ATOL
    state_match = state_diff < ATOL
    grad_match = grad_diff < ATOL

    status = "[PASS]" if (out_match and state_match and grad_match) else "[FAIL]"
    print(f"{status} Verify state gradients")
    print("=" * 60)

    assert out_match, f"Output mismatch, max diff={out_diff}"
    assert state_match, f"Final state mismatch, max diff={state_diff}"
    assert grad_match, f"Initial state gradient mismatch, max diff={grad_diff}"
    
    print("[SUCCESS] Cross-segment state gradients match perfectly!")
    print("=" * 60)

if __name__ == "__main__":
    run_verify()
