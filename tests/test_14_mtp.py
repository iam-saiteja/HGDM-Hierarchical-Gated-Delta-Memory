"""
HGDM Step 14: Multi-Token Prediction K=4 Tests
Run: python3 tests/test_14_mtp.py
Expected: ALL TESTS PASS
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
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

from hgdm_omega import OmegaConfig, OmegaGDM

print("\n[Step 14] Multi-Token Prediction K=4 Tests")
print("-" * 50)

# -- TEST 1: Logit shape and count check ---------------------------------------
def t1():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(DEVICE)
    
    x = torch.randint(0, 256, (2, 32), device=DEVICE)
    logits_list, _ = model(x, return_mtp=True)
    
    assert isinstance(logits_list, list), "Expected list of logits"
    assert len(logits_list) == 4, f"Expected 4 heads, got {len(logits_list)}"
    
    for i, logits in enumerate(logits_list):
        assert logits.shape == (2, 32, 256), f"Head {i+1} shape mismatch: {logits.shape}"
        
test("4 heads produce 4 separate logit tensors with correct shape", t1)

# -- TEST 2: Numerical loss properties -------------------------------------------
def t2():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(device=DEVICE, dtype=DTYPE)
    
    x = torch.randint(0, 256, (2, 32), device=DEVICE)
    y = torch.randint(0, 256, (2, 32), device=DEVICE)
    
    logits_list, _ = model(x, return_mtp=True)
    
    losses = []
    for k, logits in enumerate(logits_list):
        logits_f32 = logits.float()
        if k == 0:
            head_loss = F.cross_entropy(logits_f32.reshape(-1, 256), y.reshape(-1))
        else:
            head_loss = F.cross_entropy(logits_f32[:, :-k].reshape(-1, 256), y[:, k:].reshape(-1))
        losses.append(head_loss.item())
        
    print(f"    Head losses: {[f'{l:.4f}' for l in losses]}")
    for i, l in enumerate(losses):
        assert l > 0.0, f"Head {i+1} loss is not positive: {l}"
        assert not torch.isnan(torch.tensor(l)), f"Head {i+1} loss is NaN"
        
test("Each head's loss is finite and positive (>0)", t2)

# -- TEST 3: Gradient flow verification ------------------------------------------
def t3():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(device=DEVICE, dtype=DTYPE)
    
    x = torch.randint(0, 256, (2, 32), device=DEVICE)
    y = torch.randint(0, 256, (2, 32), device=DEVICE)
    
    logits_list, _ = model(x, return_mtp=True)
    
    total_loss = 0.0
    for k, logits in enumerate(logits_list):
        if k == 0:
            total_loss = total_loss + F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        else:
            total_loss = total_loss + F.cross_entropy(logits[:, :-k].reshape(-1, 256), y[:, k:].reshape(-1))
            
    total_loss.backward()
    
    embed_grad_norm = model.embedding.weight.grad.norm().item()
    print(f"    Shared embedding/head weight gradient norm: {embed_grad_norm:.6f}")
    assert embed_grad_norm > 0.0, "Weight gradient is zero"

test("Gradient flows to the shared parameters through the heads", t3)

# -- TEST 4: Head 1 loss vs Head 4 loss comparison -----------------------------
def t4():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(device=DEVICE, dtype=DTYPE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    
    # Generate a first-order Markov chain sequence where the next token is the same
    # as the current token with 90% probability.
    # This makes adjacent prediction (Head 1) significantly easier (90% predictable)
    # than 4-ahead prediction (Head 4, which is 0.9^4 ≈ 65.6% predictable).
    B, T = 4, 64
    seq = []
    for b in range(B):
        elem = []
        curr = torch.randint(0, 256, (1,)).item()
        for t in range(T):
            if torch.rand(1).item() < 0.9:
                curr = curr
            else:
                curr = torch.randint(0, 256, (1,)).item()
            elem.append(curr)
        seq.append(elem)
    pattern = torch.tensor(seq, device=DEVICE, dtype=torch.long)
    
    # Train for 40 steps to let the model learn the copy pattern
    for step in range(40):
        logits_list, _ = model(pattern[:, :-1], return_mtp=True)
        y = pattern[:, 1:]
        loss = 0.0
        for k, logits in enumerate(logits_list):
            if k == 0:
                loss = loss + F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            else:
                loss = loss + F.cross_entropy(logits[:, :-k].reshape(-1, 256), y[:, k:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    # Evaluate head losses
    with torch.no_grad():
        logits_list, _ = model(pattern[:, :-1], return_mtp=True)
        y = pattern[:, 1:]
        
    losses = []
    for k, logits in enumerate(logits_list):
        logits_f32 = logits.float()
        if k == 0:
            head_loss = F.cross_entropy(logits_f32.reshape(-1, 256), y.reshape(-1))
        else:
            head_loss = F.cross_entropy(logits_f32[:, :-k].reshape(-1, 256), y[:, k:].reshape(-1))
        losses.append(head_loss.item())
        
    print(f"    After training - Head 1 Loss: {losses[0]:.4f} | Head 4 Loss: {losses[3]:.4f}")
    assert losses[0] < losses[3], f"Expected Head 1 loss < Head 4 loss, got {losses[0]:.4f} vs {losses[3]:.4f}"

test("Adjacent prediction is easier: Head 1 loss < Head 4 loss", t4)

# -- TEST 5: Training and convergence -------------------------------------------
def t5():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(device=DEVICE, dtype=DTYPE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    losses = []
    for step in range(50):
        x = torch.randint(0, 256, (2, 32), device=DEVICE)
        y = torch.randint(0, 256, (2, 32), device=DEVICE)
        logits_list, _ = model(x, return_mtp=True)
        
        loss = 0.0
        for k, logits in enumerate(logits_list):
            if k == 0:
                loss = loss + F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            else:
                loss = loss + F.cross_entropy(logits[:, :-k].reshape(-1, 256), y[:, k:].reshape(-1))
                
        opt.zero_grad()
        loss.backward()
        assert not torch.isnan(loss), "NaN loss detected"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        
    first5 = sum(losses[:5]) / 5
    last5  = sum(losses[-5:]) / 5
    print(f"    Loss: {first5:.3f} → {last5:.3f} (improvement: {(first5-last5)/first5*100:.1f}%)")
    assert last5 < first5, f"Loss did not decrease: {first5:.3f} → {last5:.3f}"

test("Training: model trains and converges with MTP loss", t5)

print("\n" + "="*60)
fails = [k for k,v in results.items() if not v.startswith("PASS")]
if fails:
    print(f"RESULT: FAILED ({len(fails)}/{len(results)} tests failed)")
    for f in fails: print(f"  ✗ {f}\n    → {results[f]}")
    sys.exit(1)
else:
    print(f"RESULT: ALL {len(results)} TESTS PASSED ✅")
    sys.exit(0)
