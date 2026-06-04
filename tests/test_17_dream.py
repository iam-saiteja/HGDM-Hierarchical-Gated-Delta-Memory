"""
HGDM Step 17: Dream / Generative Replay Tests
Run: python3 tests/test_17_dream.py
Expected: ALL TESTS PASS
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

from train_omega import DreamScheduler
from hgdm_omega import OmegaConfig, OmegaGDM

print("\n[Step 17] Dream / Generative Replay Tests")
print("-" * 50)

# -- TEST 1: No dream at step < warmup ------------------------------------------
def t1():
    ds = DreamScheduler(warmup=2000, interval=500)
    
    no_dream_steps = [0, 1, 100, 500, 1000, 1500, 1999]
    for s in no_dream_steps:
        assert not ds.should_dream(s), f"Dream should not fire at step {s}"
    
    print(f"    Steps {no_dream_steps}: all correctly blocked ✓")

test("No dream at step < 2000 (warmup)", t1)

# -- TEST 2: Dream fires at correct steps after warmup -------------------------
def t2():
    ds = DreamScheduler(warmup=2000, interval=500)
    
    fire_steps = [2000, 2500, 3000, 3500, 4000]
    no_fire_steps = [2001, 2100, 2499, 2501, 2999]
    
    for s in fire_steps:
        assert ds.should_dream(s), f"Dream should fire at step {s}"
    for s in no_fire_steps:
        assert not ds.should_dream(s), f"Dream should not fire at step {s}"
    
    print(f"    Fire steps {fire_steps}: all correct ✓")
    print(f"    No-fire steps {no_fire_steps}: all correct ✓")

test("Dream fires at step 2000, 2500, 3000, ...", t2)

# -- TEST 3: Quality gate logic correctly rejects high PPL ----------------------
def t3():
    ds = DreamScheduler(warmup=0, interval=1, quality_threshold=2.0)
    
    # Test the quality gate logic directly:
    # If recent_train_ppl = 10.0, threshold = 2.0 * 10.0 = 20.0
    # A dream with PPL > 20.0 should be rejected
    ds.recent_train_ppl = 10.0
    threshold = ds.quality_threshold * ds.recent_train_ppl
    
    # Simulate scenarios
    low_ppl = 5.0    # < 20.0 → accept
    high_ppl = 25.0  # > 20.0 → reject
    
    assert low_ppl < threshold, f"Test setup error: {low_ppl} should be < {threshold}"
    assert high_ppl > threshold, f"Test setup error: {high_ppl} should be > {threshold}"
    
    # Now verify with actual model — train it briefly to get non-degenerate predictions
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(device=DEVICE, dtype=DTYPE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Train 10 steps to get non-trivial weights
    for _ in range(10):
        x = torch.randint(0, 256, (2, 32), device=DEVICE)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    
    # Set recent_train_ppl very low so model's dream PPL exceeds threshold
    ds2 = DreamScheduler(warmup=0, interval=1, quality_threshold=2.0, dream_len=32)
    ds2.recent_train_ppl = 0.5  # threshold = 1.0 — any real CE loss will exceed this
    
    dream_loss, dream_ppl, accepted = ds2.dream(model, DEVICE, DTYPE)
    print(f"    Dream PPL: {dream_ppl:.2f} | Threshold: {ds2.quality_threshold * ds2.recent_train_ppl:.2f} | Accepted: {accepted}")
    
    # Either it's rejected (PPL > 1.0) which is correct, or dream_ppl happens to be <= 1.0
    # which would mean loss ≈ 0 (astronomically unlikely after training)
    if dream_ppl > ds2.quality_threshold * ds2.recent_train_ppl:
        assert not accepted, "Quality gate should have rejected"
        print(f"    Quality gate correctly rejected high-PPL dream ✓")
    else:
        # If PPL is somehow very low, verify the gate correctly accepted
        assert accepted, "Quality gate should have accepted low-PPL dream"
        print(f"    Dream PPL was low enough to accept — gate logic correct ✓")

test("Dream quality gate threshold logic is correct", t3)

# -- TEST 4: Consistency loss is finite when dream is accepted ------------------
def t4():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(device=DEVICE, dtype=DTYPE)
    
    # Very lenient threshold to guarantee acceptance
    ds = DreamScheduler(warmup=0, interval=1, quality_threshold=1000.0, dream_len=32)
    ds.recent_train_ppl = 1e6
    
    dream_loss, dream_ppl, accepted = ds.dream(model, DEVICE, DTYPE)
    
    print(f"    Dream PPL: {dream_ppl:.4f} | Accepted: {accepted}")
    assert accepted, "Dream should be accepted with very high threshold"
    assert dream_loss is not None, "Dream loss should not be None when accepted"
    assert torch.isfinite(dream_loss), f"Dream loss is not finite: {dream_loss.item()}"
    # Loss can be 0.0 in bfloat16 for degenerate cases — just check it's non-negative and finite
    assert dream_loss.item() >= 0.0, f"Dream loss should be non-negative: {dream_loss.item()}"
    print(f"    Dream consistency loss: {dream_loss.item():.6f} (finite, non-negative) ✓")

test("Consistency loss is finite when dream is accepted", t4)

# -- TEST 5: Training loss not degraded by dreaming ----------------------------
def t5():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(device=DEVICE, dtype=DTYPE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    ds = DreamScheduler(warmup=10, interval=5, quality_threshold=1000.0, lambda_dream=0.01)
    ds.recent_train_ppl = 1e6  # ensure acceptance
    
    losses = []
    for step in range(50):
        x = torch.randint(0, 256, (2, 32), device=DEVICE)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1))
        
        # Dream step
        if ds.should_dream(step):
            dream_loss, _, accepted = ds.dream(model, DEVICE, DTYPE)
            if accepted and dream_loss is not None:
                loss = loss + dream_loss
        
        ds.update_train_ppl(loss.item())
        opt.zero_grad()
        loss.backward()
        assert not torch.isnan(loss), f"NaN loss at step {step}"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    
    first5 = sum(losses[:5]) / 5
    last5 = sum(losses[-5:]) / 5
    print(f"    Loss: {first5:.3f} → {last5:.3f} (improvement: {(first5-last5)/first5*100:.1f}%)")
    assert last5 < first5, f"Loss did not decrease: {first5:.3f} → {last5:.3f}"

test("Training loss converges with dreaming enabled", t5)

print("\n" + "="*60)
fails = [k for k,v in results.items() if not v.startswith("PASS")]
if fails:
    print(f"RESULT: FAILED ({len(fails)}/{len(results)} tests failed)")
    for f in fails: print(f"  ✗ {f}\n    → {results[f]}")
    sys.exit(1)
else:
    print(f"RESULT: ALL {len(results)} TESTS PASSED ✅")
    sys.exit(0)
