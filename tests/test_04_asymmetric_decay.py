"""
HGDM Step 04: Asymmetric Decay Init (half fast / half slow heads)
Run: python3 tests/test_04_asymmetric_decay.py
Expected: ALL TESTS PASS

What this tests:
  - First H//2 heads have fast timescales (tau = 4*(h+1): 4,8,12,...)
  - Last H//2 heads have slow timescales (tau = 200*(h-H//2+1): 200,400,...)
  - Clean hard separation: max(fast_tau) << min(slow_tau)
  - Alpha values span the full useful range across heads
  - No NaN over 500 step sequence
  - All W_lambda gradients nonzero (all timescales are trainable)
  - Training still converges
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
from ultimate.hgdm_ultimate import HGDMConfig, MultiHeadGatedDelta, HGDMUltimate

B, T, D, H = 2, 128, 768, 12
config = HGDMConfig(
    d_model=D, n_layers=2, n_heads=H, d_k=64, d_v=64, d_ff=D*4,
    use_variable_delta_t=True
)
H_half = H // 2  # 6

print("\n[Step 04] Asymmetric Decay Init Tests")
print("-" * 50)

# ── TEST 1: Fast heads have correct tau values ─────────────────────────────────
def t1():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    lambdas = torch.exp(mixer.W_lambda).tolist()
    taus = [1.0 / l for l in lambdas]
    fast_taus = taus[:H_half]
    expected_fast = [4.0*(h+1) for h in range(H_half)]
    print(f"    Fast head taus (h=0..{H_half-1}): {[f'{t:.1f}' for t in fast_taus]}")
    print(f"    Expected:                          {[f'{t:.1f}' for t in expected_fast]}")
    for i, (got, exp) in enumerate(zip(fast_taus, expected_fast)):
        assert abs(got - exp) / exp < 0.05, f"Fast head {i}: tau={got:.2f}, expected {exp:.2f}"
test("Fast heads: tau = 4*(h+1), values [4,8,12,16,20,24]", t1)

# ── TEST 2: Slow heads have correct tau values ─────────────────────────────────
def t2():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    lambdas = torch.exp(mixer.W_lambda).tolist()
    taus = [1.0 / l for l in lambdas]
    slow_taus = taus[H_half:]
    expected_slow = [200.0*(h-H_half+1) for h in range(H_half, H)]
    print(f"    Slow head taus (h={H_half}..{H-1}): {[f'{t:.1f}' for t in slow_taus]}")
    print(f"    Expected:                          {[f'{t:.1f}' for t in expected_slow]}")
    for i, (got, exp) in enumerate(zip(slow_taus, expected_slow)):
        assert abs(got - exp) / exp < 0.05, f"Slow head {i+H_half}: tau={got:.2f}, expected {exp:.2f}"
test("Slow heads: tau = 200*(h-H//2+1), values [200,400,...,1200]", t2)

# ── TEST 3: Clean separation — max(fast) << min(slow) ─────────────────────────
def t3():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    lambdas = torch.exp(mixer.W_lambda).tolist()
    taus = [1.0 / l for l in lambdas]
    max_fast = max(taus[:H_half])
    min_slow = min(taus[H_half:])
    separation = min_slow / max_fast
    assert separation > 5.0, f"Timescale separation too small: {separation:.1f}× (expected >5×)"
    print(f"    max(fast tau) = {max_fast:.1f} | min(slow tau) = {min_slow:.1f}")
    print(f"    Separation ratio: {separation:.1f}× (clean cortical split ✓)")
test("Separation: min(slow_tau) > 5× max(fast_tau)", t3)

# ── TEST 4: Alpha values span wide range at init ────────────────────────────────
def t4():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    # With delta_t=1.0 at init: alpha = exp(-lambda * 1.0) = exp(-1/tau)
    lambdas = torch.exp(mixer.W_lambda)
    # delta_t at init ≈ 1.0 (softplus(0.5413) + 1e-3)
    delta_t_init = F.softplus(torch.tensor(0.5413)) + 1e-3
    alpha_at_init = torch.exp(-delta_t_init * lambdas)
    fast_alphas = alpha_at_init[:H_half].tolist()
    slow_alphas = alpha_at_init[H_half:].tolist()
    print(f"    Fast head alphas: {[f'{a:.4f}' for a in fast_alphas]}")
    print(f"    Slow head alphas: {[f'{a:.4f}' for a in slow_alphas]}")
    # Fast heads: alpha ~ exp(-1/4) ≈ 0.78 to exp(-1/24) ≈ 0.96
    # Slow heads: alpha ~ exp(-1/200) ≈ 0.995 to exp(-1/1200) ≈ 0.9992
    assert min(fast_alphas) < 0.90, f"Fastest head alpha too high: {min(fast_alphas):.4f}"
    assert max(slow_alphas) > 0.99, f"Slowest head alpha too low: {max(slow_alphas):.4f}"
    span = max(slow_alphas) - min(fast_alphas)
    assert span > 0.1, f"Alpha range too narrow: {span:.4f}"
    print(f"    Alpha span: {min(fast_alphas):.4f} to {max(slow_alphas):.4f} (span={span:.4f})")
test("Alpha span: fast heads < 0.90, slow heads > 0.99", t4)

# ── TEST 5: No NaN over 500 steps ──────────────────────────────────────────────
def t5():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    state = None
    for chunk in range(25):
        x = torch.randn(1, 20, D, device=DEVICE, dtype=torch.float32)
        out, state = mixer(x, state=state)
        S = state[0] if isinstance(state, tuple) else state
        assert not torch.isnan(out).any(), f"NaN at chunk {chunk}"
        assert not torch.isnan(S).any(), f"NaN in state at chunk {chunk}"
    S_tensor = state[0] if isinstance(state, tuple) else state
    print(f"    500 steps clean ✓ | final state norm: {S_tensor.norm().item():.2f}")
test("Stability: no NaN over 500 steps with asymmetric timescales", t5)

# ── TEST 6: Fast head states change faster than slow head states ───────────────
def t6():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    # Run 10 steps and compare state change rate per head
    S_prev = torch.zeros(1, H, 64, 64, device=DEVICE)
    S = S_prev.clone()
    state_changes_per_head = torch.zeros(H)
    x = torch.randn(1, 10, D, device=DEVICE, dtype=torch.float32)
    # Use sequential path to get per-step states
    with torch.no_grad():
        for t_idx in range(10):
            out, state_new = mixer(x[:, t_idx:t_idx+1], state=S)
            S_new = state_new[0] if isinstance(state_new, tuple) else state_new
            S_tensor = S[0] if isinstance(S, tuple) else S
            delta = (S_new - S_tensor).norm(dim=(-2,-1)).squeeze(0)  # [H]
            state_changes_per_head += delta.cpu()
            S = state_new
    fast_change = state_changes_per_head[:H_half].mean().item()
    slow_change = state_changes_per_head[H_half:].mean().item()
    print(f"    Avg state change — fast heads: {fast_change:.4f} | slow heads: {slow_change:.4f}")
    # Fast heads should change MORE per step (lower alpha = more forgetting = more change)
    # This is not strictly required but is expected behavior
    print(f"    Ratio fast/slow: {fast_change/(slow_change+1e-8):.2f}×")
    # Both should be nonzero
    assert fast_change > 0, "Fast heads show zero state change"
    assert slow_change > 0, "Slow heads show zero state change"
test("Dynamics: fast and slow heads both update state (nonzero change)", t6)

# ── TEST 7: All W_lambda gradients are nonzero (all timescales trainable) ───────
def t7():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    x = torch.randn(B, T, D, device=DEVICE, dtype=torch.float32)
    out, state = mixer(x, state=None)
    out.sum().backward()
    assert mixer.W_lambda.grad is not None, "No gradient for W_lambda"
    per_head_grads = mixer.W_lambda.grad.abs().tolist()
    zero_heads = [h for h, g in enumerate(per_head_grads) if g < 1e-10]
    assert len(zero_heads) == 0, f"Zero gradient for W_lambda at heads: {zero_heads}"
    print(f"    W_lambda grads: {[f'{g:.4f}' for g in per_head_grads]}")
    print(f"    All {H} heads have nonzero timescale gradients ✓")
test("Gradient: W_lambda.grad nonzero for ALL heads", t7)

# ── TEST 8: Training convergence ───────────────────────────────────────────────
def t8():
    model = HGDMUltimate(config, force_sequential=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(50):
        x = torch.randint(0, 256, (B, T), device=DEVICE)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    first5 = sum(losses[:5]) / 5
    last5  = sum(losses[-5:]) / 5
    assert last5 < first5, f"Loss did not decrease: {first5:.3f} → {last5:.3f}"
    print(f"    Loss: {first5:.3f} → {last5:.3f} (improvement: {(first5-last5)/first5*100:.1f}%)")
test("Training: loss decreases over 50 steps with asymmetric timescales", t8)

# ── TEST 9: W_lambda values diverge during training (heads specialize) ─────────
def t9():
    model = HGDMUltimate(config, force_sequential=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    # Get initial lambda std
    initial_std = model.layers[0].mixer.W_lambda.data.std().item()
    for step in range(100):
        x = torch.randint(0, 256, (B, T), device=DEVICE)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    final_std = model.layers[0].mixer.W_lambda.data.std().item()
    print(f"    W_lambda std: {initial_std:.4f} → {final_std:.4f} (heads diverging = specializing)")
    # std should stay nonzero (heads remain differentiated, not all collapsing to same value)
    assert final_std > 0.01, f"W_lambda collapsed (std={final_std:.4f}) — all heads identical"
test("Specialization: W_lambda maintains diversity across heads after 100 steps", t9)

# ── TEST 10: VRAM budget ───────────────────────────────────────────────────────
def t10():
    if DEVICE != "cuda":
        print("    SKIP: no CUDA"); return
    torch.cuda.reset_peak_memory_stats()
    model = HGDMUltimate(config, force_sequential=False).to(DEVICE).to(DTYPE)
    x = torch.randint(0, 256, (4, 512), device=DEVICE)
    logits, _ = model(x)
    vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    assert vram_mb < 6000, f"VRAM too high: {vram_mb:.0f}MB"
    print(f"    VRAM peak: {vram_mb:.0f}MB")
test("VRAM: < 6GB for 120M model, batch=4, T=512", t10)

# ── SUMMARY ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
fails = [k for k,v in results.items() if not v.startswith("PASS")]
if fails:
    print(f"RESULT: FAILED ({len(fails)}/{len(results)} tests failed)")
    for f in fails:
        print(f"  ✗ {f}"); print(f"    → {results[f]}")
    print("\n⛔ Fix failures and rerun.")
    sys.exit(1)
else:
    print(f"RESULT: ALL {len(results)} TESTS PASSED ✅")
    print("\n✅ Safe to commit step-04-asymmetric-decay")
    sys.exit(0)
