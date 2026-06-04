"""
HGDM Step 11: Rotary Position Embeddings (RoPE)
Run: python3 tests/test_11_rope.py
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

from ultimate.hgdm_ultimate import HGDMConfig, RoPEEmbedding
from hgdm_omega import OmegaConfig, OmegaGDM

print("\n[Step 11] Rotary Position Embedding (RoPE) Tests")
print("-" * 50)

# -- TEST 1: RoPE Rotary Property Invariance -------------------------------------
def t1():
    dim = 64
    rope = RoPEEmbedding(dim=dim, max_position_embeddings=1000).to(DEVICE)
    
    q = torch.randn(1, 1, 1, dim, device=DEVICE)
    k = torch.randn(1, 1, 1, dim, device=DEVICE)
    
    # Helper to apply RoPE to a single vector at position pos
    def apply_rope_at_pos(x, pos):
        cos, sin = rope(x, seq_len=1, offset=pos)
        # Apply rotation
        d = x.shape[-1]
        x1 = x[..., :d // 2]
        x2 = x[..., d // 2:]
        rotated_x = torch.cat((-x2, x1), dim=-1)
        return x * cos[None, :, None, :] + rotated_x * sin[None, :, None, :]

    # We check three pairs where t - s = 3:
    # 1. t=5, s=2
    # 2. t=10, s=7
    # 3. t=103, s=100
    pairs = [(5, 2), (10, 7), (103, 100)]
    dots = []
    
    for t, s in pairs:
        q_rot = apply_rope_at_pos(q, t)
        k_rot = apply_rope_at_pos(k, s)
        dot = torch.sum(q_rot * k_rot).item()
        dots.append(dot)
        
    print(f"    Dot products for relative distance 3: {dots}")
    assert abs(dots[0] - dots[1]) < 1e-4, f"Pair 1 vs 2 mismatch: {dots[0]} vs {dots[1]}"
    assert abs(dots[0] - dots[2]) < 1e-4, f"Pair 1 vs 3 mismatch: {dots[0]} vs {dots[2]}"
    print("    Relative distance dot products are invariant ✓")
test("Relative position invariance: <f(q, t), f(k, s)> depends only on t - s", t1)

# -- TEST 2: VRAM / Parameter check ----------------------------------------------
def t2():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(DEVICE)
    
    # Verify semantic_pos_embed is absent
    assert not hasattr(model, 'semantic_pos_embed'), "Absolute semantic_pos_embed parameter was not removed!"
    
    # Verify core layers configuration has use_rope enabled
    for i, layer in enumerate(model.semantic_core):
        assert layer.mixer.use_rope == True, f"Layer {i} mixer does not use RoPE"
        
    print("    Absolute positional embedding removed successfully ✓")
test("Parameter removal: semantic_pos_embed removed and use_rope enabled in core layers", t2)

# -- TEST 3: Infinite-length generalization check -------------------------------
def t3():
    # Verify that generation at length > max_position_embeddings does not crash
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512,
        max_position_embeddings=128
    )
    model = OmegaGDM(config).to(DEVICE)
    
    # Pass seq of length 16 with offset 200 (which exceeds max_position_embeddings)
    x = torch.randint(0, 256, (1, 16), device=DEVICE)
    
    try:
        logits, states = model(x, offset=200)
        assert logits.shape == (1, 16, 256), f"Output shape mismatch: {logits.shape}"
        print("    Forward pass with offset > max_position_embeddings succeeded without index out of bounds ✓")
    except Exception as e:
        raise AssertionError(f"Forward pass failed: {e}")
test("Infinite-length check: generation at offset > max_position_embeddings does not crash", t3)

# -- TEST 4: Training & Convergence check -----------------------------------------
def t4():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    
    losses = []
    for step in range(50):
        x = torch.randint(0, 256, (2, 32), device=DEVICE)
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
test("Training: model converges successfully with RoPE in semantic core", t4)

print("\n" + "="*60)
fails = [k for k,v in results.items() if not v.startswith("PASS")]
if fails:
    print(f"RESULT: FAILED ({len(fails)}/{len(results)} tests failed)")
    for f in fails: print(f"  ✗ {f}\n    → {results[f]}")
    sys.exit(1)
else:
    print(f"RESULT: ALL {len(results)} TESTS PASSED ✅")
    sys.exit(0)
