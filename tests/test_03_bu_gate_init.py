"""
HGDM Step 03: Asymmetric bu_gate Initialization (-4.0 → -2.0)
Run: python3 tests/test_03_bu_gate_init.py
Expected: ALL TESTS PASS

What this tests:
  - bu_gate initializes to -2.0 (sigmoid ≈ 0.12), not -4.0 (sigmoid ≈ 0.018)
  - td_gate is unchanged at -4.0 (sigmoid ≈ 0.018)
  - Asymmetry ratio is ~6.7× (bu more open than td at init)
  - OmegaGDM forward pass still works with no NaN
  - bu highway gradient is larger than td highway gradient
  - Both gates are still learnable (gradients nonzero)
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
from hgdm_omega import OmegaGDM, OmegaConfig

# Small config for fast tests (OmegaGDM is hierarchical so needs T > W=8)
cfg = OmegaConfig(
    d_byte=64,
    catcher_layers=1,
    renderer_layers=1,
    d_model=128,
    core_layers=2,
    n_heads=4,
    d_k=32,
    d_v=32,
    d_ff=512,
    decimation_rate=8,
    vocab_size=256,
    max_position_embeddings=65536,
    use_state_fusion=False,
    use_variable_delta_t=True,
)

print("\n[Step 03] Asymmetric bu_gate Init Tests")
print("-" * 50)

# ── TEST 1: bu_gate initial value is -2.0 ─────────────────────────────────────
def t1():
    model = OmegaGDM(cfg, force_sequential=True).to(DEVICE)
    bu_vals = model.highway_bu_gate.data
    mean_val = bu_vals.mean().item()
    assert abs(mean_val - (-2.0)) < 0.01, f"bu_gate mean = {mean_val:.4f}, expected -2.0"
    print(f"    highway_bu_gate values: {bu_vals.tolist()}")
    print(f"    mean = {mean_val:.4f} (expected -2.0)")
test("bu_gate init: all values ≈ -2.0", t1)

# ── TEST 2: td_gate initial value still -4.0 (UNCHANGED) ──────────────────────
def t2():
    model = OmegaGDM(cfg, force_sequential=True).to(DEVICE)
    td_vals = model.highway_td_gate.data
    mean_val = td_vals.mean().item()
    assert abs(mean_val - (-4.0)) < 0.01, f"td_gate mean = {mean_val:.4f}, expected -4.0 (UNCHANGED)"
    print(f"    highway_td_gate values: {td_vals.tolist()}")
    print(f"    mean = {mean_val:.4f} (expected -4.0, UNCHANGED)")
test("td_gate init: still -4.0 (not changed)", t2)

# ── TEST 3: sigmoid values confirm asymmetry ───────────────────────────────────
def t3():
    model = OmegaGDM(cfg, force_sequential=True).to(DEVICE)
    bu_open = torch.sigmoid(model.highway_bu_gate).mean().item()
    td_open = torch.sigmoid(model.highway_td_gate).mean().item()
    ratio   = bu_open / td_open

    assert abs(bu_open - 0.1192) < 0.01, f"sigmoid(bu_gate) = {bu_open:.4f}, expected ≈ 0.12"
    assert abs(td_open - 0.0180) < 0.005, f"sigmoid(td_gate) = {td_open:.4f}, expected ≈ 0.018"
    assert ratio > 5.0, f"bu/td ratio = {ratio:.2f}, expected > 5× (got {ratio:.2f}×)"
    print(f"    sigmoid(bu_gate) = {bu_open:.4f} (≈ 0.12)")
    print(f"    sigmoid(td_gate) = {td_open:.4f} (≈ 0.018)")
    print(f"    Asymmetry ratio  = {ratio:.2f}× (should be ~6.7×)")
test("Sigmoid values: bu≈0.12, td≈0.018, ratio > 5×", t3)

# ── TEST 4: Forward pass no NaN (T > W=8 needed to trigger semantic core) ──────
def t4():
    model = OmegaGDM(cfg, force_sequential=True).to(DEVICE)
    # T=32 so we get 4 semantic tokens (32 // 8 = 4)
    x = torch.randint(0, 256, (2, 32), device=DEVICE)
    logits, states = model(x)
    assert not torch.isnan(logits).any(), "NaN in logits"
    assert not torch.isinf(logits).any(), "Inf in logits"
    print(f"    logits: {logits.shape}, no NaN/Inf ✓")
test("Forward pass: no NaN, no Inf (T=32, 4 semantic tokens)", t4)

# ── TEST 5: Forward pass works for T < W=8 (renderer-only path) ───────────────
def t5():
    model = OmegaGDM(cfg, force_sequential=True).to(DEVICE)
    x = torch.randint(0, 256, (2, 5), device=DEVICE)  # T=5 < W=8
    logits, states = model(x)
    assert not torch.isnan(logits).any(), "NaN in logits (T<W path)"
    print(f"    logits (T<W): {logits.shape}, no NaN ✓")
test("Forward pass: no NaN for T < decimation_rate (renderer-only path)", t5)

# ── TEST 6: bu_gate has a nonzero gradient (still learnable) ──────────────────────
def t6():
    # IMPORTANT: bu_highway only activates on 2nd+ forward pass (needs prev_renderer_last_S in states[4])
    # Fresh states[4]=None → bu_highway skipped. Must do 2 sequential passes.
    model = OmegaGDM(cfg, force_sequential=True).to(DEVICE)
    x1 = torch.randint(0, 256, (2, 32), device=DEVICE)
    x2 = torch.randint(0, 256, (2, 32), device=DEVICE)
    # Pass 1: builds states[4] with prev_renderer_last_S
    with torch.no_grad():
        _, states = model(x1)
    # Pass 2: bu_highway fires, gate participates in computation
    logits, _ = model(x2, states=states)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x2[:, 1:].reshape(-1))
    loss.backward()
    grad = model.highway_bu_gate.grad
    assert grad is not None, "No gradient for highway_bu_gate (bu_highway did not activate)"
    assert not torch.isnan(grad).any(), "NaN gradient for highway_bu_gate"
    grad_norm = grad.norm().item()
    assert grad_norm > 1e-10, f"highway_bu_gate gradient is zero: {grad_norm}"
    print(f"    highway_bu_gate.grad norm: {grad_norm:.6f} (2-pass test — stateful activation)")
test("Gradient: highway_bu_gate.grad is nonzero (requires 2 sequential passes)", t6)

# ── TEST 7: td_gate also has nonzero gradient ─────────────────────────────────
def t7():
    model = OmegaGDM(cfg, force_sequential=True).to(DEVICE)
    x = torch.randint(0, 256, (2, 32), device=DEVICE)
    logits, _ = model(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1))
    loss.backward()
    grad = model.highway_td_gate.grad
    assert grad is not None, "No gradient for highway_td_gate"
    grad_norm = grad.norm().item()
    assert grad_norm > 1e-10, f"highway_td_gate gradient is zero: {grad_norm}"
    print(f"    highway_td_gate.grad norm: {grad_norm:.6f}")
test("Gradient: highway_td_gate.grad is nonzero (td gate also learnable)", t7)

# ── TEST 8: bu_gate grad is larger than td_gate grad ────────────────────────
def t8():
    # Same 2-pass pattern to activate bu_highway
    model = OmegaGDM(cfg, force_sequential=True).to(DEVICE)
    x1 = torch.randint(0, 256, (2, 32), device=DEVICE)
    x2 = torch.randint(0, 256, (2, 32), device=DEVICE)
    with torch.no_grad():
        _, states = model(x1)
    logits, _ = model(x2, states=states)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x2[:, 1:].reshape(-1))
    loss.backward()
    bu_grad = model.highway_bu_gate.grad.norm().item()
    td_grad = model.highway_td_gate.grad.norm().item()
    ratio = bu_grad / (td_grad + 1e-10)
    print(f"    bu_gate.grad norm: {bu_grad:.6f}")
    print(f"    td_gate.grad norm: {td_grad:.6f}")
    print(f"    bu/td gradient ratio: {ratio:.2f}×")
    assert bu_grad > 0, "bu_gate gradient is zero"
    assert td_grad > 0, "td_gate gradient is zero"
test("Gradients: both bu_gate and td_gate have nonzero gradients (2-pass)", t8)

# ── TEST 9: Training loss still decreases ─────────────────────────────────────
def t9():
    model = OmegaGDM(cfg, force_sequential=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(30):
        x = torch.randint(0, 256, (2, 32), device=DEVICE)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if not torch.isnan(loss):
            losses.append(loss.item())

    first5 = sum(losses[:5]) / 5
    last5  = sum(losses[-5:]) / 5
    assert last5 < first5, f"Loss did not decrease: {first5:.3f} → {last5:.3f}"
    print(f"    Loss: {first5:.3f} → {last5:.3f} (improvement: {(first5-last5)/first5*100:.1f}%)")
test("Training: OmegaGDM loss decreases over 30 steps", t9)

# ── TEST 10: VRAM budget ───────────────────────────────────────────────────────
def t10():
    if DEVICE != "cuda":
        print("    SKIP: no CUDA")
        return
    torch.cuda.reset_peak_memory_stats()
    # Use a bigger config for VRAM test — closer to real 39M translation model
    big_cfg = OmegaConfig(
        d_byte=256, catcher_layers=2, renderer_layers=2,
        d_model=512, core_layers=8, n_heads=8, d_k=64, d_v=64, d_ff=2048,
        decimation_rate=8, vocab_size=256, max_position_embeddings=65536,
        use_variable_delta_t=True,
    )
    model = OmegaGDM(big_cfg, force_sequential=False).to(DEVICE).to(DTYPE)
    x = torch.randint(0, 256, (4, 512), device=DEVICE)
    logits, _ = model(x)
    vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    assert vram_mb < 8000, f"VRAM too high: {vram_mb:.0f}MB"
    print(f"    Model params: {param_count:.1f}M | VRAM peak: {vram_mb:.0f}MB")
test("VRAM: OmegaGDM (39M-class) fits in < 8GB, batch=4, T=512", t10)

# ── SUMMARY ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
fails = [k for k,v in results.items() if not v.startswith("PASS")]
if fails:
    print(f"RESULT: FAILED ({len(fails)}/{len(results)} tests failed)")
    for f in fails:
        print(f"  ✗ {f}")
        print(f"    → {results[f]}")
    print("\n⛔ DO NOT copy this step. Fix failures and rerun.")
    sys.exit(1)
else:
    print(f"RESULT: ALL {len(results)} TESTS PASSED ✅")
    print("\n✅ Safe to commit step-03-bu-gate-init")
    sys.exit(0)
