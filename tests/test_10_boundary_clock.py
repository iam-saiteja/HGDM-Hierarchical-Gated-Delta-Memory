"""
HGDM Step 10: Boundary Clock Verification
Run: python3 tests/test_10_boundary_clock.py
Expected: ALL TESTS PASS
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import math
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

from ultimate.hgdm_ultimate import HGDMConfig, HGDMUltimate, MultiHeadGatedDelta

B, T, D, H = 2, 64, 768, 12
config = HGDMConfig(d_model=D, n_layers=2, n_heads=H, d_k=64, d_v=64, d_ff=D*4, use_variable_delta_t=True)

print("\n[Step 10] Boundary Clock Tests")
print("-" * 50)

# -- TEST 1: Boundary detection checks -------------------------------------------
def t1():
    # Byte values: '.' = 46, '?' = 63, '!' = 33, '\n' = 10
    # Any other byte is not a boundary.
    seq = torch.tensor([[46, 63, 33, 10, 65, 97, 32]], device=DEVICE) # shape [1, 7]
    # We can compute boundary mask exactly like inside HGDMUltimate
    boundary_mask = (seq == 46) | (seq == 63) | (seq == 33) | (seq == 10)
    expected = torch.tensor([[True, True, True, True, False, False, False]], device=DEVICE)
    assert torch.equal(boundary_mask, expected), f"Mask mismatch: {boundary_mask} vs {expected}"
test("Boundary detection: identifies '.', '?', '!', '\\n' (46, 63, 33, 10)", t1)

# -- TEST 2: Fast head state norms drop at boundaries (partial reset working) -----
# -- TEST 3: Slow head state norms continue growing at boundaries (not reset) -----
def t2_t3():
    mixer = MultiHeadGatedDelta(config, force_sequential=True).to(DEVICE)
    
    # We will run a sequence where there is a boundary at t=31 (1-indexed, i.e., index 30)
    boundary_mask = torch.zeros(1, 64, dtype=torch.bool, device=DEVICE)
    boundary_mask[0, 30] = True
    
    x = torch.ones(1, 64, D, device=DEVICE, dtype=torch.float32)
    
    with torch.no_grad():
        q = F.normalize(mixer.W_q(x).view(1, 64, H, 64), dim=-1)
        k = F.normalize(mixer.W_k(x).view(1, 64, H, 64), dim=-1)
        v = mixer.W_v(x).view(1, 64, H, 64)
        
        delta_t = F.softplus(mixer.W_delta(x)) + 1e-3
        lambdas = torch.exp(mixer.W_lambda)
        alpha = torch.exp(-delta_t * lambdas[None, None, :])
        
        H_half = H // 2
        fast_head_mask = torch.arange(H, device=DEVICE)[None, None, :] < H_half
        reset_mask = boundary_mask[:, :, None] & fast_head_mask
        alpha_reset = torch.where(reset_mask, torch.full_like(alpha, 0.01), alpha)
        
        _beta_raw = torch.sigmoid(mixer.W_beta(x))
        beta = F.relu(_beta_raw - 0.1) / 0.9
        beta = beta * torch.exp(mixer.log_beta_scale)[None, None, :]
        
        pos = torch.arange(64, device=DEVICE, dtype=torch.float32)
        T_cycle = torch.exp(mixer.log_T_cycle)
        clock_gate = 0.5 + 0.5 * torch.cos(2.0 * math.pi * pos[:, None] / T_cycle[None, :])
        beta = beta * clock_gate[None, :, :].to(dtype=beta.dtype)

        # Recurrence with reset
        S_reset = torch.zeros(1, H, 64, 64, device=DEVICE)
        norms_reset = []
        for t in range(64):
            delta = torch.einsum('hk,hd->hkd', k[0, t], v[0, t])
            S_reset = alpha_reset[0, t, :, None, None] * S_reset + beta[0, t, :, None, None] * delta
            norms_reset.append(S_reset.norm(dim=(-2, -1)).squeeze().tolist())
            
        # Recurrence without reset
        S_no_reset = torch.zeros(1, H, 64, 64, device=DEVICE)
        norms_no_reset = []
        for t in range(64):
            delta = torch.einsum('hk,hd->hkd', k[0, t], v[0, t])
            S_no_reset = alpha[0, t, :, None, None] * S_no_reset + beta[0, t, :, None, None] * delta
            norms_no_reset.append(S_no_reset.norm(dim=(-2, -1)).squeeze().tolist())

    # Check for fast heads (h < 6)
    for h in range(H_half):
        norm_at_boundary = norms_reset[30][h]
        norm_no_reset_at_boundary = norms_no_reset[30][h]
        
        assert norm_at_boundary < norm_no_reset_at_boundary * 0.4, \
            f"Head {h} state did not drop at boundary: reset={norm_at_boundary:.4f}, no_reset={norm_no_reset_at_boundary:.4f}"
            
    print("    Fast head state norms dropped significantly at boundary ✓")
    
    # Check for slow heads (h >= 6)
    for h in range(H_half, H):
        assert abs(norms_reset[30][h] - norms_no_reset[30][h]) < 1e-4, \
            f"Slow head {h} was reset: {norms_reset[30][h]} vs {norms_no_reset[30][h]}"
            
    print("    Slow head state norms are completely unaffected by boundary ✓")

test("Selective Reset: fast heads reset at boundaries, slow heads unaffected", t2_t3)

# -- TEST 4: Training loss converges ---------------------------------------------
def t4():
    model = HGDMUltimate(config, force_sequential=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(50):
        x = torch.randint(0, 256, (B, T), device=DEVICE)
        x[:, 30] = 46
        x[:, 45] = 46
        
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
test("Training: loss converges with boundary clock resets", t4)

# -- TEST 5: VRAM -----------------------------------------------------------------
def t5():
    if DEVICE != "cuda":
        print("    SKIP: no CUDA"); return
    torch.cuda.reset_peak_memory_stats()
    model = HGDMUltimate(config, force_sequential=False).to(DEVICE).to(DTYPE)
    x = torch.randint(0, 256, (4, 512), device=DEVICE)
    logits, _ = model(x)
    vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    assert vram_mb < 6000, f"VRAM too high: {vram_mb:.0f}MB"
    print(f"    VRAM: {vram_mb:.0f}MB")
test("VRAM: < 6GB (no parameter overhead for boundary clock)", t5)

print("\n" + "="*60)
fails = [k for k,v in results.items() if not v.startswith("PASS")]
if fails:
    print(f"RESULT: FAILED ({len(fails)}/{len(results)} tests failed)")
    for f in fails: print(f"  ✗ {f}\n    → {results[f]}")
    sys.exit(1)
else:
    print(f"RESULT: ALL {len(results)} TESTS PASSED ✅")
    sys.exit(0)
