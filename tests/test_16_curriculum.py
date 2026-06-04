"""
HGDM Step 16: Self-Organizing Curriculum Tests
Run: python3 tests/test_16_curriculum.py
Expected: ALL TESTS PASS
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import traceback
import time

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

from train_omega import SelfOrganizingCurriculum

print("\n[Step 16] Self-Organizing Curriculum Tests")
print("-" * 50)

# -- TEST 1: Higher loss documents sampled more frequently ----------------------
def t1():
    cur = SelfOrganizingCurriculum(n_docs=10, alpha=0.5, temp=3.0)
    
    # Simulate: doc 0 has high loss, doc 9 has low loss
    for _ in range(20):
        cur.update(0, 5.0)   # hard document
        cur.update(9, 0.5)   # easy document
        for i in range(1, 9):
            cur.update(i, 1.0)  # medium documents
    
    # Sample 1000 times and count
    counts = torch.zeros(10)
    for _ in range(1000):
        idx = cur.sample(k=1)[0]
        counts[idx] += 1
        
    print(f"    Doc 0 (hard) sampled: {counts[0]:.0f}/1000 | Doc 9 (easy) sampled: {counts[9]:.0f}/1000")
    assert counts[0] > counts[9], f"Hard doc not sampled more: {counts[0]:.0f} vs {counts[9]:.0f}"
    assert counts[0] > 100, f"Hard doc sampled too rarely: {counts[0]:.0f}"

test("Higher loss documents sampled more frequently", t1)

# -- TEST 2: Distribution is not uniform (KL divergence > 0.01) ----------------
def t2():
    cur = SelfOrganizingCurriculum(n_docs=10, alpha=0.5, temp=3.0)
    
    # Update with varying losses
    for _ in range(20):
        for i in range(10):
            cur.update(i, float(i + 1))  # loss = 1, 2, ..., 10
    
    kl = cur.kl_vs_uniform()
    print(f"    KL divergence vs uniform: {kl:.6f}")
    assert kl > 0.01, f"KL divergence too small: {kl}"
    
    # Also verify probabilities sum to 1
    prob_sum = cur.probs.sum().item()
    assert abs(prob_sum - 1.0) < 1e-5, f"Probabilities don't sum to 1: {prob_sum}"

test("Curriculum distribution is non-uniform (KL > 0.01)", t2)

# -- TEST 3: Curriculum sampling overhead < 1ms --------------------------------
def t3():
    cur = SelfOrganizingCurriculum(n_docs=1000, alpha=0.1, temp=2.0)
    
    # Warm up with some updates
    for i in range(1000):
        cur.update(i, torch.rand(1).item() * 5.0)
    
    # Measure time for 1000 samples
    t0 = time.time()
    for _ in range(1000):
        cur.sample(k=1)
    elapsed = (time.time() - t0) / 1000 * 1000  # ms per sample
    
    print(f"    Sampling time: {elapsed:.4f} ms/sample")
    assert elapsed < 1.0, f"Sampling too slow: {elapsed:.4f} ms"

test("Sampling overhead < 1ms per step", t3)

# -- TEST 4: EMA tracking and update correctness --------------------------------
def t4():
    cur = SelfOrganizingCurriculum(n_docs=5, alpha=0.3, temp=2.0)
    
    # All start at 1.0
    assert torch.allclose(cur.ema_losses, torch.ones(5)), "Initial EMA not all 1.0"
    
    # Update doc 0 with loss=10.0
    cur.update(0, 10.0)
    expected = (1 - 0.3) * 1.0 + 0.3 * 10.0  # = 3.7
    assert abs(cur.ema_losses[0].item() - expected) < 1e-5, f"EMA update wrong: {cur.ema_losses[0].item()} vs {expected}"
    
    # Doc 0 should now have highest probability
    assert cur.probs[0] > cur.probs[1], "Doc with highest loss should have highest prob"
    print(f"    After update: doc0 EMA={cur.ema_losses[0].item():.4f}, prob={cur.probs[0].item():.4f}")

test("EMA tracking and probability update are correct", t4)

print("\n" + "="*60)
fails = [k for k,v in results.items() if not v.startswith("PASS")]
if fails:
    print(f"RESULT: FAILED ({len(fails)}/{len(results)} tests failed)")
    for f in fails: print(f"  ✗ {f}\n    → {results[f]}")
    sys.exit(1)
else:
    print(f"RESULT: ALL {len(results)} TESTS PASSED ✅")
    sys.exit(0)
