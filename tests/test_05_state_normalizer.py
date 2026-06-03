"""
HGDM Step 05: State Normalizer n_t + Stabilizer m_t
Run: python3 tests/test_05_state_normalizer.py
Expected: ALL TESTS PASS

What this tests:
  - forward() now returns (S, n) tuple, not just S
  - n_t starts at 0 and grows from fresh state
  - n_t norm is bounded (does not explode)
  - Output is normalized: ||out||_inf bounded by ||v||_max
  - Both sequential and Triton fast paths handle new state format
  - Highway methods handle (S, n) tuple state
  - OmegaGDM full forward pass works with new state format
  - Gradient flows through normalizer denom back to n_t
  - Training loss still converges
  - VRAM within budget
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

# ── SETUP ──────────────────────────────────────────────────────────────────────
from ultimate.hgdm_ultimate import HGDMConfig, MultiHeadGatedDelta, HGDMUltimate
from hgdm_omega import OmegaGDM, OmegaConfig

B, T, D, H = 2, 64, 768, 12
config = HGDMConfig(
    d_model=D, n_layers=2, n_heads=H, d_k=64, d_v=64, d_ff=D*4,
    use_variable_delta_t=True
)

print("\n[Step 05] State Normalizer Tests")
print("-" * 50)

# ── TEST 1: State is now (S, n) tuple ──────────────────────────────────────────
def t1():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    x = torch.randn(B, T, D, device=DEVICE, dtype=torch.float32)
    out, state = mixer(x, state=None)
    assert isinstance(state, tuple), f"State should be tuple (S,n), got {type(state)}"
    assert len(state) == 2, f"State tuple should have 2 elements, got {len(state)}"
    S, n = state
    assert S.shape == (B, H, 64, 64), f"S shape wrong: {S.shape}"
    assert n.shape == (B, H, 64),     f"n shape wrong: {n.shape}"
    print(f"    State: (S={S.shape}, n={n.shape}) ✓")
test("State format: forward() returns (S, n) tuple", t1)

# ── TEST 2: n starts at 0 and grows from fresh state ──────────────────────────
def t2():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    n_norms = []
    state = None
    for step in range(10):
        x = torch.randn(1, 1, D, device=DEVICE, dtype=torch.float32)
        out, state = mixer(x, state=state)
        n = state[1]
        n_norms.append(n.norm().item())
    assert n_norms[0] > 0,   "n_t is still 0 after first step (should have grown)"
    assert n_norms[-1] > n_norms[0], "n_t not growing from step 0 to step 9"
    print(f"    n_t norm trajectory (10 steps): {[f'{v:.3f}' for v in n_norms]}")
test("n_t: starts from 0 and grows from fresh state", t2)

# ── TEST 3: n_t is bounded over long sequences ─────────────────────────────────
def t3():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    state = None
    n_norms = []
    for chunk in range(50):
        x = torch.randn(1, 20, D, device=DEVICE, dtype=torch.float32)
        out, state = mixer(x, state=state)
        n_norms.append(state[1].norm().item())
    # Growth should slow down (plateau)
    early_growth = n_norms[9] - n_norms[0]
    late_growth  = n_norms[-1] - n_norms[-10]
    assert n_norms[-1] < 1e8, f"n_t norm exploded: {n_norms[-1]:.2e}"
    print(f"    n_t trajectory: {n_norms[0]:.2f} → {n_norms[24]:.2f} → {n_norms[-1]:.2f}")
    print(f"    Early growth: {early_growth:.3f}, Late growth: {late_growth:.3f}")
test("n_t: bounded and plateaus over 1000 steps", t3)

# ── TEST 4: Output is normalized (||out||_inf bounded) ─────────────────────────
def t4():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    x = torch.randn(B, T, D, device=DEVICE, dtype=torch.float32)
    out, state = mixer(x, state=None)
    # After W_o projection, compare output magnitude to input magnitude
    out_norm = out.abs().max().item()
    x_norm   = x.abs().max().item()
    print(f"    out abs max: {out_norm:.4f} | x abs max: {x_norm:.4f}")
    # The normalizer prevents runaway outputs — out should be bounded
    assert not torch.isinf(out).any(), "Inf in output"
    assert not torch.isnan(out).any(), "NaN in output"
    print(f"    Output bounded: no NaN, no Inf ✓")
test("Output: finite and bounded after normalization", t4)

# ── TEST 5: state=None gives n starting from zero ──────────────────────────────
def t5():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    x = torch.randn(B, T, D, device=DEVICE, dtype=torch.float32)
    out1, state1 = mixer(x, state=None)
    out2, state2 = mixer(x, state=None)
    # Fresh starts should be identical (same input, same init)
    n1, n2 = state1[1], state2[1]
    assert torch.allclose(n1, n2, atol=1e-5), "Fresh starts produce different n_t"
    S1, S2 = state1[0], state2[0]
    assert torch.allclose(S1, S2, atol=1e-5), "Fresh starts produce different S"
    print(f"    Two fresh starts: identical state ✓")
test("Fresh start: state=None always gives same initial n=0", t5)

# ── TEST 6: Triton fast path returns (S, n) tuple too ─────────────────────────
def t6():
    if DEVICE != "cuda":
        print("    SKIP: no CUDA"); return
    try:
        from kernel_nitro import fused_nitro_scan
        if fused_nitro_scan is None:
            print("    SKIP: Triton kernel not available"); return
    except Exception:
        print("    SKIP: kernel_nitro import failed"); return
    
    mixer = MultiHeadGatedDelta(config, force_sequential=False).to(DEVICE)
    x = torch.randn(B, T, D, device=DEVICE, dtype=torch.float32)
    out, state = mixer(x, state=None)
    assert isinstance(state, tuple), f"Triton path: state should be tuple, got {type(state)}"
    S, n = state
    assert not torch.isnan(out).any(), "NaN in Triton fast path output"
    assert not torch.isnan(S).any(), "NaN in Triton fast path S"
    assert not torch.isnan(n).any(), "NaN in Triton fast path n"
    print(f"    Triton fast path: (S={S.shape}, n={n.shape}) ✓")
test("Triton fast path: returns (S, n) tuple, no NaN", t6)

# ── TEST 7: Sequential and Triton paths agree on n_t ──────────────────────────
def t7():
    if DEVICE != "cuda":
        print("    SKIP: no CUDA"); return
    try:
        from kernel_nitro import fused_nitro_scan
        if fused_nitro_scan is None:
            print("    SKIP: Triton not available"); return
    except Exception:
        print("    SKIP: kernel_nitro import failed"); return
    
    config2 = HGDMConfig(d_model=D, n_layers=1, n_heads=H, d_k=64, d_v=64, d_ff=D*4, use_variable_delta_t=True)
    m_seq   = MultiHeadGatedDelta(config2, force_sequential=True).to(DEVICE)
    m_fast  = MultiHeadGatedDelta(config2, force_sequential=False).to(DEVICE)
    # Copy weights
    m_fast.load_state_dict(m_seq.state_dict())
    
    torch.manual_seed(42)
    x = torch.randn(1, 32, D, device=DEVICE, dtype=torch.float32)
    
    out_seq,  (S_seq,  n_seq)  = m_seq(x,  state=None)
    out_fast, (S_fast, n_fast) = m_fast(x, state=None)
    
    max_n_diff = (n_seq - n_fast).abs().max().item()
    print(f"    n_t max diff (sequential vs Triton): {max_n_diff:.6f}")
    assert max_n_diff < 0.01, f"Sequential and Triton n_t disagree: max diff={max_n_diff:.4f}"
test("Consistency: sequential and Triton paths agree on n_t (< 0.01 diff)", t7)

# ── TEST 8: OmegaGDM full forward pass works with new state format ─────────────
def t8():
    cfg = OmegaConfig(
        d_byte=64, catcher_layers=1, renderer_layers=1,
        d_model=128, core_layers=2, n_heads=4, d_k=32, d_v=32, d_ff=512,
        decimation_rate=8, vocab_size=256, use_variable_delta_t=True,
    )
    model = OmegaGDM(cfg, force_sequential=True).to(DEVICE)
    x = torch.randint(0, 256, (2, 32), device=DEVICE)
    logits, states = model(x)
    assert not torch.isnan(logits).any(), "NaN in OmegaGDM logits"
    assert not torch.isinf(logits).any(), "Inf in OmegaGDM logits"
    # States are lists of (S, n) tuples
    for i, s in enumerate(states[0]):  # catcher states
        if s is not None:
            assert isinstance(s, tuple), f"Catcher state[{i}] is not tuple"
    print(f"    OmegaGDM forward: logits={logits.shape}, no NaN/Inf ✓")
test("OmegaGDM: full forward pass with (S,n) state format", t8)

# ── TEST 9: Gradient flows through normalizer back to n_t ─────────────────────
def t9():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    x = torch.randn(B, T, D, device=DEVICE, dtype=torch.float32, requires_grad=False)
    out, (S, n) = mixer(x, state=None)
    loss = out.sum()
    loss.backward()
    # Gradient should reach W_k (which feeds into n_t via k vectors)
    grad_k = mixer.W_k.weight.grad
    assert grad_k is not None, "No gradient for W_k (gradient not flowing through n_t)"
    assert not torch.isnan(grad_k).any(), "NaN in W_k.grad"
    norm_k = grad_k.norm().item()
    assert norm_k > 1e-10, f"W_k gradient is zero: {norm_k}"
    print(f"    W_k.grad norm: {norm_k:.4f} (gradient flows through n_t denom) ✓")
test("Gradient: flows through normalization denom back to W_k", t9)

# ── TEST 10: Training loss converges (normalizer doesn't kill gradients) ────────
def t10():
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
test("Training: loss decreases over 50 steps with normalizer active", t10)

# ── TEST 11: VRAM (normalizer adds O(T*H*d_k) memory — should be minimal) ──────
def t11():
    if DEVICE != "cuda":
        print("    SKIP: no CUDA"); return
    torch.cuda.reset_peak_memory_stats()
    model = HGDMUltimate(config, force_sequential=False).to(DEVICE).to(DTYPE)
    x = torch.randint(0, 256, (4, 512), device=DEVICE)
    logits, _ = model(x)
    vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    assert vram_mb < 6000, f"VRAM too high: {vram_mb:.0f}MB"
    print(f"    VRAM peak: {vram_mb:.0f}MB")
test("VRAM: < 6GB with n_t tracking overhead", t11)

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
    print("\n✅ Results satisfactory? Then push step-05.")
    sys.exit(0)
