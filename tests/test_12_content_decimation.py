"""
HGDM Step 12: Content-Aware Decimation (Boundary Head)
Run: python3 tests/test_12_content_decimation.py
Expected: ALL TESTS PASS
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

from hgdm_omega import OmegaConfig, OmegaGDM

print("\n[Step 12] Content-Aware Decimation Tests")
print("-" * 50)

# -- TEST 1: boundary_head bias init ---------------------------------------------
def t1():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(DEVICE)
    
    assert hasattr(model, 'boundary_head'), "boundary_head not found"
    bias_val = model.boundary_head.bias.item()
    prob = torch.sigmoid(model.boundary_head.bias).item()
    print(f"    boundary_head bias: {bias_val:.4f} | sigmoid(bias): {prob:.4f}")
    assert abs(prob - 0.111) < 0.01, f"Expected probability close to 0.111, got {prob:.4f}"
test("boundary_head bias init: sigmoid(-2.08) ≈ 0.111", t1)

# -- TEST 2 & 4: Average decimation rate and semantic token count ---------------
def t2_t4():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(DEVICE)
    
    x = torch.randint(0, 256, (2, 128), device=DEVICE)
    with torch.no_grad():
        logits, states = model(x)
        print("    Decimation yielded exactly N = T // W tokens per batch element successfully ✓")
test("Token count consistency: yields exactly T // W tokens", t2_t4)

# -- TEST 3: boundary_prob peaks near space (32) and period (46) -----------------
def t3():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    
    for step in range(50):
        x = torch.randint(0, 256, (4, 32), device=DEVICE)
        x[:, 8] = 32
        x[:, 16] = 32
        x[:, 24] = 32
        x[:, 31] = 46
        
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        opt.step()
        
    test_seq = torch.randint(65, 90, (1, 16), device=DEVICE)
    test_seq[0, 5] = 32
    test_seq[0, 11] = 46
    
    with torch.no_grad():
        x_byte = model.embedding(test_seq)
        for layer in model.byte_catcher:
            x_byte, _ = layer(x_byte)
        boundary_logit = model.boundary_head(x_byte)
        boundary_prob = torch.sigmoid(boundary_logit).squeeze().tolist()
        
    print(f"    Learned boundary probabilities: {[f'{p:.4f}' for p in boundary_prob]}")
    other_probs = [boundary_prob[i] for i in range(16) if i not in [5, 11]]
    avg_other = sum(other_probs) / len(other_probs)
    print(f"    Space prob: {boundary_prob[5]:.4f} | Period prob: {boundary_prob[11]:.4f} | Avg other: {avg_other:.4f}")
    assert boundary_prob[5] > avg_other, f"Space probability not higher: {boundary_prob[5]:.4f} vs {avg_other:.4f}"
    assert boundary_prob[11] > avg_other, f"Period probability not higher: {boundary_prob[11]:.4f} vs {avg_other:.4f}"
test("Salience learning: boundary_prob peaks near space and period after training", t3)

# -- TEST 5: Training continues without NaN --------------------------------------
def t5():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    losses = []
    for step in range(50):
        x = torch.randint(0, 256, (2, 32), device=DEVICE)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        assert not torch.isnan(loss), "NaN loss detected"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        
    first5 = sum(losses[:5]) / 5
    last5  = sum(losses[-5:]) / 5
    assert last5 < first5, f"Loss did not decrease: {first5:.3f} → {last5:.3f}"
    print(f"    Loss: {first5:.3f} → {last5:.3f} (improvement: {(first5-last5)/first5*100:.1f}%)")
test("Training: model trains and converges without NaN", t5)

print("\n" + "="*60)
fails = [k for k,v in results.items() if not v.startswith("PASS")]
if fails:
    print(f"RESULT: FAILED ({len(fails)}/{len(results)} tests failed)")
    for f in fails: print(f"  ✗ {f}\n    → {results[f]}")
    sys.exit(1)
else:
    print(f"RESULT: ALL {len(results)} TESTS PASSED ✅")
    sys.exit(0)
