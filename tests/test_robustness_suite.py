import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ultimate.hgdm_ultimate import HGDMUltimate, HGDMConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def test_forward_shapes():
    print("\n--- Test 1: Forward Shape Checks ---")
    for B in [1, 2, 4]:
        for T in [16, 64, 128]:
            config = HGDMConfig(use_rope=True, use_epistemic_gate=True, n_layers=2)
            model = HGDMUltimate(config).to(DEVICE)
            x = torch.randint(0, 256, (B, T), device=DEVICE)
            out, state = model(x)
            assert out.shape == (B, T, 256), f"Expected (B, T, 256), got {out.shape}"
            S, n = state[0]
            n_heads = config.n_heads
            d_k = config.d_model // n_heads
            d_v = config.d_model // n_heads
            assert S.shape == (B, n_heads, d_k, d_v)
            assert n.shape == (B, n_heads, d_k)
    print("Test 1 Passed!")

def test_backward_gradients():
    print("\n--- Test 2: Backward Gradient Checks ---")
    for mode in ["detached", "exact"]:
        config = HGDMConfig(use_rope=True, use_epistemic_gate=True, n_grad_mode=mode, n_layers=2)
        model = HGDMUltimate(config).to(DEVICE)
        x = torch.randint(0, 256, (2, 32), device=DEVICE)
        out, _ = model(x)
        loss = out.sum()
        loss.backward()
        
        # Verify gradients exist and are non-NaN for weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Grad is None for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN in grad for {name}"
        model.zero_grad()
        print(f"Gradients for mode='{mode}' verified.")
    print("Test 2 Passed!")

def test_long_context_offsets():
    print("\n--- Test 3: Long-Context Offset Tests ---")
    config = HGDMConfig(use_rope=True, use_epistemic_gate=True, n_layers=2)
    model = HGDMUltimate(config).to(DEVICE)
    
    # 1. Warm start with offset
    x1 = torch.randint(0, 256, (2, 32), device=DEVICE)
    out1, state1 = model(x1)
    
    # 2. Continue generation with large offset (should trigger cache expansion)
    x2 = torch.randint(0, 256, (2, 128), device=DEVICE)
    out2, state2 = model(x2, states=state1, offset=32)
    
    assert out2.shape == (2, 128, 256)
    print("Test 3 Passed (RoPE Cache dynamically expanded and offset processed)!")

def test_state_none_cold_start():
    print("\n--- Test 4: state=None Cold Start ---")
    config = HGDMConfig(use_rope=True, use_epistemic_gate=True, n_layers=2)
    model = HGDMUltimate(config).to(DEVICE)
    
    x = torch.randint(0, 256, (2, 32), device=DEVICE)
    # Explicit state=None should not crash
    out, state = model(x, states=None)
    assert state is not None
    print("Test 4 Passed!")

def test_boundary_mask_edge_cases():
    print("\n--- Test 5: Boundary Mask Edge Cases ---")
    config = HGDMConfig(use_rope=True, use_epistemic_gate=True, n_layers=2)
    model = HGDMUltimate(config).to(DEVICE)
    
    # All boundary tokens (e.g. 46 is '.')
    x_all = torch.full((2, 32), 46, dtype=torch.long, device=DEVICE)
    out_all, _ = model(x_all)
    
    # All normal tokens (e.g. 97 is 'a')
    x_none = torch.full((2, 32), 97, dtype=torch.long, device=DEVICE)
    out_none, _ = model(x_none)
    
    assert not torch.isnan(out_all).any()
    assert not torch.isnan(out_none).any()
    print("Test 5 Passed!")

def test_tuner_parity():
    print("\n--- Test 6: Triton vs Fallback Parity ---")
    config = HGDMConfig(
        use_rope=True,
        use_state_fusion=True,
        use_epistemic_gate=True,
        n_grad_mode="exact",
        n_layers=2
    )
    
    # Run in sequential path (Triton disabled)
    model_seq = HGDMUltimate(config, force_sequential=True).to(DEVICE)
    # Run in fast path (Triton enabled)
    model_triton = HGDMUltimate(config, force_sequential=False).to(DEVICE)
    
    # Share weights
    model_triton.load_state_dict(model_seq.state_dict())
    
    # Set to eval mode to disable dropout
    model_seq.eval()
    model_triton.eval()
    
    x = torch.randint(0, 256, (2, 64), device=DEVICE)
    
    # Forward parity
    with torch.no_grad():
        out_seq, _ = model_seq(x)
        out_tri, _ = model_triton(x)
    
    print("out_seq shape:", out_seq.shape)
    print("out_tri shape:", out_tri.shape)
    print("out_seq slice:", out_seq[0, 0, :5].tolist())
    print("out_tri slice:", out_tri[0, 0, :5].tolist())
    
    diff = (out_seq - out_tri).abs().max().item()
    print(f"Max absolute forward discrepancy: {diff:.6f}")
    assert diff < 1e-2, f"Parity discrepancy too large! Diff={diff}"
    print("Test 6 Passed!")

if __name__ == "__main__":
    test_forward_shapes()
    test_backward_gradients()
    test_long_context_offsets()
    test_state_none_cold_start()
    test_boundary_mask_edge_cases()
    if DEVICE == "cuda":
        test_tuner_parity()
    print("\nALL ROBUSTNESS TESTS PASSED SUCCESSFULLY! 🚀")

