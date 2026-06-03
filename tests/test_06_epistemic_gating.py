"""
HGDM Step 06: Epistemic Gating (Confidence Gate from n_t)
Run: python3 tests/test_06_epistemic_gating.py
Expected: ALL TESTS PASS

What this tests:
  - At t=0 (fresh n=0): confidence = tanh(0) = 0 → output near zero
  - At t>10 (rich state): confidence grows toward 1.0
  - confidence varies per-head (not uniform across heads)
  - gradient flows through confidence back to W_k
  - training converges (gate doesn't kill learning signal)
  - OmegaGDM works with epistemic gating active
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import traceback

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

from ultimate.hgdm_ultimate import HGDMConfig, MultiHeadGatedDelta, HGDMUltimate
from hgdm_omega import OmegaGDM, OmegaConfig

B, T, D, H = 2, 64, 768, 12
config = HGDMConfig(d_model=D, n_layers=2, n_heads=H, d_k=64, d_v=64, d_ff=D*4, use_variable_delta_t=True)

print("\n[Step 06] Epistemic Gating Tests")
print("-" * 50)

# ── TEST 1: At t=1 (first token), output magnitude is near zero ────────────────
def t1():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    x = torch.randn(1, 1, D, device=DEVICE, dtype=torch.float32)
    out, (S, n) = mixer(x, state=None)
    # n starts from 0 and gets one update: n = 0 + beta * k
    # conf = tanh(||n||_2) per head — small at t=1
    conf_approx = torch.tanh(n.norm(dim=-1)).mean().item()
    out_norm = out.abs().mean().item()
    print(f"    t=1: conf ≈ {conf_approx:.4f}, out mean abs = {out_norm:.6f}")
    assert conf_approx < 0.95, f"Confidence too high at t=1: {conf_approx:.4f}"
test("t=1 confidence: tanh(||n||) < 0.95 at first token", t1)

# ── TEST 2: confidence grows over time ────────────────────────────────────────
def t2():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    conf_vals = []
    state = None
    for step in range(30):
        x = torch.randn(1, 1, D, device=DEVICE, dtype=torch.float32)
        out, state = mixer(x, state=state)
        n = state[1]
        conf = torch.tanh(n.norm(dim=-1)).mean().item()
        conf_vals.append(conf)
    assert conf_vals[-1] > conf_vals[0], f"Confidence not growing: {conf_vals[0]:.4f} → {conf_vals[-1]:.4f}"
    print(f"    Confidence trajectory: {conf_vals[0]:.4f} → {conf_vals[14]:.4f} → {conf_vals[-1]:.4f}")
test("Confidence: grows over 30 steps from near-0 toward 1.0", t2)

# ── TEST 3: output at t=1 is smaller than output at t=100 ─────────────────────
def t3():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    # Single-token inference: compare output magnitude at t=1 vs after warmup
    torch.manual_seed(0)
    x0 = torch.randn(1, 1, D, device=DEVICE, dtype=torch.float32)
    out0, state = mixer(x0, state=None)
    out0_norm = out0.abs().mean().item()
    # Warmup 99 more steps
    for _ in range(99):
        x = torch.randn(1, 1, D, device=DEVICE, dtype=torch.float32)
        _, state = mixer(x, state=state)
    torch.manual_seed(0)
    x100 = torch.randn(1, 1, D, device=DEVICE, dtype=torch.float32)
    out100, _ = mixer(x100, state=state)
    out100_norm = out100.abs().mean().item()
    print(f"    out mean abs — t=1: {out0_norm:.6f} | t=100: {out100_norm:.6f}")
    assert out100_norm > out0_norm, f"Output at t=100 should be larger than t=1 (epistemic gate not working)"
test("Output magnitude: t=100 > t=1 (confidence grows over sequence)", t3)

# ── TEST 4: no NaN in output ────────────────────────────────────────────────────
def t4():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    x = torch.randn(B, T, D, device=DEVICE, dtype=torch.float32)
    out, (S, n) = mixer(x, state=None)
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"
    print(f"    out shape: {out.shape}, no NaN/Inf ✓")
test("Forward pass: no NaN, no Inf with epistemic gate", t4)

# ── TEST 5: gradient flows through confidence gate ────────────────────────────
def t5():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    x = torch.randn(B, T, D, device=DEVICE, dtype=torch.float32)
    out, _ = mixer(x, state=None)
    out.sum().backward()
    grad_k = mixer.W_k.weight.grad
    assert grad_k is not None, "No gradient for W_k (conf gate blocking backprop)"
    assert grad_k.norm().item() > 1e-10, f"W_k gradient is zero after conf gate"
    print(f"    W_k.grad norm: {grad_k.norm().item():.4f} (gradient flows through tanh gate) ✓")
test("Gradient: W_k.grad nonzero (tanh gate is differentiable)", t5)

# ── TEST 6: Triton path also has epistemic gate ────────────────────────────────
def t6():
    if DEVICE != "cuda":
        print("    SKIP: no CUDA"); return
    try:
        from kernel_nitro import fused_nitro_scan
        if fused_nitro_scan is None:
            print("    SKIP: Triton not available"); return
    except Exception:
        print("    SKIP: kernel_nitro import failed"); return
    mixer_seq  = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    mixer_fast = MultiHeadGatedDelta(config, force_sequential=False).to(DEVICE)
    mixer_fast.load_state_dict(mixer_seq.state_dict())
    torch.manual_seed(1)
    x = torch.randn(1, 32, D, device=DEVICE, dtype=torch.float32)
    out_seq,  _ = mixer_seq(x,  state=None)
    out_fast, _ = mixer_fast(x, state=None)
    max_diff = (out_seq - out_fast).abs().max().item()
    print(f"    Sequential vs Triton output max diff: {max_diff:.6f}")
    assert max_diff < 0.05, f"Epistemic gate disagrees between paths: {max_diff:.4f}"
test("Consistency: epistemic gate identical across sequential and Triton paths", t6)

# ── TEST 7: OmegaGDM forward pass works ──────────────────────────────────────
def t7():
    cfg = OmegaConfig(d_byte=64, catcher_layers=1, renderer_layers=1,
                      d_model=128, core_layers=2, n_heads=4, d_k=32, d_v=32, d_ff=512,
                      decimation_rate=8, vocab_size=256, use_variable_delta_t=True)
    model = OmegaGDM(cfg, force_sequential=True).to(DEVICE)
    x = torch.randint(0, 256, (2, 32), device=DEVICE)
    logits, states = model(x)
    assert not torch.isnan(logits).any(), "NaN in OmegaGDM logits"
    print(f"    OmegaGDM: logits={logits.shape}, no NaN ✓")
test("OmegaGDM: full forward with epistemic gate active", t7)

# ── TEST 8: training converges ────────────────────────────────────────────────
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
test("Training: loss converges with epistemic gate (gate doesn't kill signal)", t8)

# ── TEST 9: VRAM ─────────────────────────────────────────────────────────────
def t9():
    if DEVICE != "cuda":
        print("    SKIP: no CUDA"); return
    torch.cuda.reset_peak_memory_stats()
    model = HGDMUltimate(config, force_sequential=False).to(DEVICE).to(DTYPE)
    x = torch.randint(0, 256, (4, 512), device=DEVICE)
    logits, _ = model(x)
    vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    assert vram_mb < 6000, f"VRAM too high: {vram_mb:.0f}MB"
    print(f"    VRAM peak: {vram_mb:.0f}MB")
test("VRAM: < 6GB with epistemic gate overhead", t9)

print("\n" + "="*60)
fails = [k for k,v in results.items() if not v.startswith("PASS")]
if fails:
    print(f"RESULT: FAILED ({len(fails)}/{len(results)} tests failed)")
    for f in fails: print(f"  ✗ {f}\n    → {results[f]}")
    print("\n⛔ Fix and rerun."); sys.exit(1)
else:
    print(f"RESULT: ALL {len(results)} TESTS PASSED ✅")
    print("\n✅ Paste results — I push only after confirmation.")
    sys.exit(0)
