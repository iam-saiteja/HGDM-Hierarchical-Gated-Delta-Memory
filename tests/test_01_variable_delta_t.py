"""
HGDM Step 01: Variable Δt ON (Time-Based Model Switch)
Run: python tests/test_01_variable_delta_t.py
Expected: ALL TESTS PASS
What this tests:
  - W_delta produces positive delta_t via softplus
  - alpha values are in (0,1) via exp(-positive)
  - Gradient flows through W_delta (model can learn when to forget)
  - Training loss decreases over 50 steps
  - VRAM within budget
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import traceback
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32
print(f"Device: {DEVICE} | Dtype: {DTYPE}")

results = {}

def test(name, fn):
    try:
        fn()
        results[name] = "PASS"
        print(f"  ✅ {name}")
    except AssertionError as e:
        results[name] = f"FAIL: {e}"
        print(f"  ❌ {name}: {e}")
    except Exception as e:
        results[name] = f"ERROR: {e}"
        print(f"  💥 {name}: {e}")
        traceback.print_exc()

# ── SETUP ──────────────────────────────────────────────────────────────────────

from ultimate.hgdm_ultimate import HGDMConfig, MultiHeadGatedDelta, HGDMLayer, HGDMUltimate

B, T, D, H = 2, 128, 768, 12
config = HGDMConfig(
    d_model=D, n_layers=2, n_heads=H, d_k=64, d_v=64, d_ff=D*4,
    use_variable_delta_t=True   # ← THE THING WE ARE TESTING
)

print("\n[Step 01] Variable Δt Tests")
print("-" * 50)

# ── TEST 1: Flag is recognized ─────────────────────────────────────────────────

def t1():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    assert hasattr(mixer, 'W_delta'), "W_delta not found on mixer"
    assert hasattr(mixer, 'W_lambda'), "W_lambda not found on mixer"
    assert not hasattr(mixer, 'W_alpha'), "W_alpha should NOT exist when use_variable_delta_t=True"
test("Config: W_delta and W_lambda exist, W_alpha absent", t1)

# ── TEST 2: delta_t is positive ────────────────────────────────────────────────

def t2():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    x = torch.randn(B, T, D, device=DEVICE, dtype=torch.float32)
    delta_t = F.softplus(mixer.W_delta(x)) + 1e-3   # same formula as forward()
    assert delta_t.min().item() > 0, f"delta_t has non-positive values: min={delta_t.min()}"
    assert delta_t.shape == (B, T, H), f"delta_t shape wrong: {delta_t.shape}"
test("delta_t: all positive, correct shape", t2)

# ── TEST 3: alpha in (0,1) ─────────────────────────────────────────────────────

def t3():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    x = torch.randn(B, T, D, device=DEVICE, dtype=torch.float32)
    delta_t = F.softplus(mixer.W_delta(x)) + 1e-3
    lambdas = torch.exp(mixer.W_lambda)
    alpha = torch.exp(-delta_t * lambdas[None, None, :])
    assert alpha.min().item() >= 0.0, f"alpha < 0: {alpha.min()}"
    assert alpha.max().item() <= 1.0, f"alpha > 1: {alpha.max()}"
    print(f"    alpha range: [{alpha.min():.4f}, {alpha.max():.4f}]")
test("alpha: all in [0,1]", t3)

# ── TEST 4: No NaN or Inf in forward pass ──────────────────────────────────────

def t4():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    x = torch.randn(B, T, D, device=DEVICE, dtype=torch.float32)
    out, state = mixer(x, state=None)
    S = state[0] if isinstance(state, tuple) else state
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"
    assert not torch.isnan(S).any(), "NaN in state"
    print(f"    out shape: {out.shape}, S shape: {S.shape}")
test("Forward pass: no NaN, no Inf", t4)

# ── TEST 5: Gradient flows through W_delta ────────────────────────────────────

def t5():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    x = torch.randn(B, T, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
    out, state = mixer(x, state=None)
    S = state[0] if isinstance(state, tuple) else state
    loss = out.sum()
    loss.backward()
    assert mixer.W_delta.weight.grad is not None, "No gradient for W_delta.weight"
    assert not torch.isnan(mixer.W_delta.weight.grad).any(), "NaN gradient for W_delta"
    grad_norm = mixer.W_delta.weight.grad.norm().item()
    assert grad_norm > 1e-10, f"W_delta gradient is zero (norm={grad_norm})"
    print(f"    W_delta.weight.grad norm: {grad_norm:.6f}")
test("Gradient: W_delta.weight.grad is nonzero and finite", t5)

# ── TEST 6: W_lambda gradients (per-head decay rates trainable) ───────────────

def t6():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    x = torch.randn(B, T, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
    out, state = mixer(x, state=None)
    S = state[0] if isinstance(state, tuple) else state
    loss = out.sum()
    loss.backward()
    assert mixer.W_lambda.grad is not None, "No gradient for W_lambda"
    grad_norm = mixer.W_lambda.grad.norm().item()
    assert grad_norm > 1e-10, f"W_lambda gradient is zero: {grad_norm}"
    print(f"    W_lambda.grad norm: {grad_norm:.6f}")
test("Gradient: W_lambda.grad is nonzero (per-head decay learns)", t6)

# ── TEST 7: Initialization — delta_t ≈ 1.0 at step 0 ─────────────────────────

def t7():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    x = torch.zeros(1, 1, D, device=DEVICE, dtype=torch.float32)  # zero input
    delta_t = F.softplus(mixer.W_delta(x)) + 1e-3
    # W_delta.weight is 0, W_delta.bias is 0.5413 → softplus(0.5413) ≈ 1.0
    expected = 1.0 + 1e-3
    actual = delta_t.mean().item()
    assert abs(actual - expected) < 0.1, f"delta_t at zero input should be ≈{expected}, got {actual}"
    print(f"    delta_t at zero input: {actual:.4f} (expected ≈ {expected:.4f})")
test("Initialization: delta_t ≈ 1.0 at zero input (bias=0.5413)", t7)

# ── TEST 8: Different content → different delta_t (input-dependent) ───────────

def t8():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    # After a few training steps, W_delta.weight should be nonzero
    # Force it to be nonzero to test input-dependence
    with torch.no_grad():
        mixer.W_delta.weight.normal_(0, 0.1)
    x1 = torch.randn(1, 1, D, device=DEVICE, dtype=torch.float32)
    x2 = torch.randn(1, 1, D, device=DEVICE, dtype=torch.float32)
    dt1 = F.softplus(mixer.W_delta(x1)) + 1e-3
    dt2 = F.softplus(mixer.W_delta(x2)) + 1e-3
    assert not torch.allclose(dt1, dt2), "delta_t is identical for different inputs (not input-dependent)"
    diff = (dt1 - dt2).abs().mean().item()
    print(f"    delta_t difference between inputs: {diff:.4f}")
test("Input-dependence: different content → different delta_t", t8)

# ── TEST 9: Training loss decreases over 50 steps ─────────────────────────────

def t9():
    model = HGDMUltimate(config, force_sequential=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(50):
        x = torch.randint(0, 256, (B, T), device=DEVICE)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    first5 = sum(losses[:5]) / 5
    last5  = sum(losses[-5:]) / 5
    assert last5 < first5, f"Loss did not decrease: first5={first5:.3f}, last5={last5:.3f}"
    print(f"    Loss: {first5:.3f} → {last5:.3f} (improvement: {(first5-last5)/first5*100:.1f}%)")
test("Training: loss decreases over 50 steps", t9)

# ── TEST 10: VRAM usage ────────────────────────────────────────────────────────

def t10():
    if DEVICE != "cuda":
        print("    SKIP: no CUDA")
        return
    torch.cuda.reset_peak_memory_stats()
    model = HGDMUltimate(config, force_sequential=False).to(DEVICE).to(DTYPE)
    x = torch.randint(0, 256, (4, 512), device=DEVICE)
    logits, _ = model(x)
    vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    assert vram_mb < 6000, f"VRAM too high: {vram_mb:.0f}MB (limit: 6000MB)"
    print(f"    VRAM peak: {vram_mb:.0f}MB")
test("VRAM: < 6GB for 120M model, batch=4, T=512", t10)

# ── SUMMARY ────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
fails = [k for k,v in results.items() if not v.startswith("PASS")]
if fails:
    print(f"RESULT: FAILED ({len(fails)}/{len(results)} tests failed)")
    for f in fails:
        print(f"  ✗ {f}")
        print(f"    → {results[f]}")
    print("\n⛔ DO NOT push this step. Fix failures and rerun.")
    sys.exit(1)
else:
    print(f"RESULT: ALL {len(results)} TESTS PASSED ✅")
    print("\n✅ Safe to: git add -A && git commit -m 'feat(step-01): variable delta_t ON — all tests passed'")
    sys.exit(0)
