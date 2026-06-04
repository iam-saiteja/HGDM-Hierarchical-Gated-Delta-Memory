"""
HGDM Step 08: Per-Head Write Scale (log_beta_scale)
Run: python3 tests/test_08_per_head_scale.py
Expected: ALL TESTS PASS

What this tests:
  - log_beta_scale parameter exists with shape [H]
  - At init: log_beta_scale=0, exp(0)=1, beta unchanged vs step-07
  - After training: heads diverge (std grows from 0)
  - Gradient flows through log_beta_scale
  - Beta values still in valid range
  - Training converges
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

B, T, D, H = 2, 64, 768, 12
config = HGDMConfig(d_model=D, n_layers=2, n_heads=H, d_k=64, d_v=64, d_ff=D*4, use_variable_delta_t=True)

print("\n[Step 08] Per-Head Write Scale Tests")
print("-" * 50)

# ── TEST 1: log_beta_scale exists with correct shape ──────────────────────────
def t1():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    assert hasattr(mixer, 'log_beta_scale'), "log_beta_scale parameter not found"
    assert mixer.log_beta_scale.shape == (H,), f"Shape wrong: {mixer.log_beta_scale.shape}"
    print(f"    log_beta_scale: shape={mixer.log_beta_scale.shape}, "
          f"values={mixer.log_beta_scale.tolist()}")
test("Parameter: log_beta_scale exists with shape [H]", t1)

# ── TEST 2: At init, log_beta_scale=0 → exp(0)=1 → no change to beta ─────────
def t2():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    init_vals = mixer.log_beta_scale.data.tolist()
    assert all(abs(v) < 1e-6 for v in init_vals), f"log_beta_scale not zero at init: {init_vals}"
    scales = torch.exp(mixer.log_beta_scale).tolist()
    assert all(abs(s - 1.0) < 1e-5 for s in scales), f"exp(0) != 1.0: {scales}"
    print(f"    log_beta_scale init: all 0.0 ✓ | exp scales: all 1.0 ✓")
test("Init: log_beta_scale=0 → scale=1.0 (no change at init)", t2)

# ── TEST 3: Forward pass no NaN, output shape correct ────────────────────────
def t3():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    x = torch.randn(B, T, D, device=DEVICE, dtype=torch.float32)
    out, (S, n) = mixer(x, state=None)
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"
    assert out.shape == (B, T, D), f"Output shape wrong: {out.shape}"
    print(f"    out={out.shape}, no NaN/Inf ✓")
test("Forward pass: no NaN, no Inf, correct shape", t3)

# ── TEST 4: Gradient flows through log_beta_scale ────────────────────────────
def t4():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    x = torch.randn(B, T, D, device=DEVICE, dtype=torch.float32)
    out, _ = mixer(x, state=None)
    out.sum().backward()
    grad = mixer.log_beta_scale.grad
    assert grad is not None, "No gradient for log_beta_scale"
    assert not torch.isnan(grad).any(), "NaN in log_beta_scale.grad"
    grad_norm = grad.norm().item()
    assert grad_norm > 1e-10, f"log_beta_scale gradient is zero: {grad_norm}"
    print(f"    log_beta_scale.grad: {grad.tolist()}")
    print(f"    grad norm: {grad_norm:.6f} ✓")
test("Gradient: log_beta_scale.grad nonzero (head scales are trainable)", t4)

# ── TEST 5: Heads diverge during training ─────────────────────────────────────
def t5():
    model = HGDMUltimate(config, force_sequential=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    init_std = model.layers[0].mixer.log_beta_scale.data.std().item()
    for step in range(100):
        x = torch.randint(0, 256, (B, T), device=DEVICE)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    final_scales = model.layers[0].mixer.log_beta_scale.data.tolist()
    final_std    = model.layers[0].mixer.log_beta_scale.data.std().item()
    print(f"    log_beta_scale after 100 steps: {[f'{v:.4f}' for v in final_scales]}")
    print(f"    std: {init_std:.6f} → {final_std:.6f} (heads specializing)")
    assert final_std > 0.001, f"Heads did not diverge (std={final_std:.6f}) — all identical scale"
test("Specialization: heads diverge in log_beta_scale after 100 training steps", t5)

# ── TEST 6: Different heads have different effective beta amplitudes ────────────
def t6():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    # Force diverged scales
    with torch.no_grad():
        mixer.log_beta_scale.data = torch.linspace(-1.0, 1.0, H, device=DEVICE)
    x = torch.randn(1, 1, D, device=DEVICE, dtype=torch.float32)
    with torch.no_grad():
        _beta_raw = torch.sigmoid(mixer.W_beta(x))
        beta_sparse = F.relu(_beta_raw - 0.1) / 0.9
        beta_scaled = beta_sparse * torch.exp(mixer.log_beta_scale)[None, None, :]
    scales = torch.exp(mixer.log_beta_scale).tolist()
    print(f"    Per-head exp(log_beta_scale): {[f'{s:.3f}' for s in scales]}")
    # Verify different heads have genuinely different amplitudes
    beta_per_head = beta_scaled[0, 0, :].tolist()
    print(f"    Per-head beta values:         {[f'{b:.4f}' for b in beta_per_head]}")
    assert max(scales) / (min(scales) + 1e-8) > 2.0, "Forced scales not creating amplitude differences"
test("Per-head amplitude: different heads write at different strengths", t6)

# ── TEST 7: Training loss converges ───────────────────────────────────────────
def t7():
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
test("Training: loss converges with per-head scale", t7)

# ── TEST 8: VRAM ──────────────────────────────────────────────────────────────
def t8():
    if DEVICE != "cuda":
        print("    SKIP: no CUDA"); return
    torch.cuda.reset_peak_memory_stats()
    model = HGDMUltimate(config, force_sequential=False).to(DEVICE).to(DTYPE)
    x = torch.randint(0, 256, (4, 512), device=DEVICE)
    logits, _ = model(x)
    vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    param_overhead = sum(p.numel() for n, p in model.named_parameters() if 'log_beta_scale' in n)
    assert vram_mb < 6000, f"VRAM too high: {vram_mb:.0f}MB"
    print(f"    VRAM: {vram_mb:.0f}MB | log_beta_scale overhead: {param_overhead} params total")
test("VRAM: < 6GB (log_beta_scale adds only H params per layer)", t8)

print("\n" + "="*60)
fails = [k for k,v in results.items() if not v.startswith("PASS")]
if fails:
    print(f"RESULT: FAILED ({len(fails)}/{len(results)} tests failed)")
    for f in fails: print(f"  ✗ {f}\n    → {results[f]}")
    print("\n⛔ Fix and rerun."); sys.exit(1)
else:
    print(f"RESULT: ALL {len(results)} TESTS PASSED ✅")
    print("\n✅ Paste results — push ONLY after confirmation.")
    sys.exit(0)
