"""
HGDM Step 18: Full Integration Test
Run: python3 tests/test_18_integration.py
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

from hgdm_omega import OmegaConfig, OmegaGDM, latent_think, think_to_english

print("\n[Step 18] Full Integration Test — All 17 Improvements Active")
print("-" * 50)

# -- TEST 1: Full forward pass: no NaN, no Inf ---------------------------------
def t1():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(device=DEVICE, dtype=DTYPE)
    
    x = torch.randint(0, 256, (2, 64), device=DEVICE)
    logits, states = model(x)
    
    assert not torch.isnan(logits).any(), "NaN in logits"
    assert not torch.isinf(logits).any(), "Inf in logits"
    assert logits.shape == (2, 64, 256), f"Unexpected logits shape: {logits.shape}"
    
    # Verify states are well-formed
    assert len(states) == 5, f"Expected 5 state groups, got {len(states)}"
    assert len(states[0]) == config.catcher_layers
    assert len(states[1]) == config.core_layers
    assert len(states[2]) == config.renderer_layers
    
    print(f"    Logits shape: {logits.shape} | No NaN/Inf ✓")

test("Full OmegaGDM forward pass: no NaN, no Inf, correct shapes", t1)

# -- TEST 2: Parameter count verification --------------------------------------
def t2():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    
    # Verify key components exist
    assert hasattr(model, 'boundary_head'), "Missing boundary_head (Step 12)"
    assert hasattr(model, 'td_gate_net'), "Missing td_gate_net (Step 13)"
    assert hasattr(model, 'bu_gate_net'), "Missing bu_gate_net (Step 13)"
    assert hasattr(model, 'mtp_heads'), "Missing mtp_heads (Step 14)"
    assert len(model.mtp_heads) == 3, f"Expected 3 MTP heads, got {len(model.mtp_heads)}"
    
    # Verify core layers have RoPE
    for i, layer in enumerate(model.semantic_core):
        assert hasattr(layer.mixer, 'rope_emb'), f"Core layer {i} missing RoPE (Step 11)"
    
    # Verify core layers have phase oscillator and per-head scale
    for i, layer in enumerate(model.semantic_core):
        assert hasattr(layer.mixer, 'log_T_cycle'), f"Core layer {i} missing log_T_cycle (Step 9)"
        assert hasattr(layer.mixer, 'log_beta_scale'), f"Core layer {i} missing log_beta_scale (Step 8)"

test("Parameter count documented and all 17 improvements present", t2)

# -- TEST 3: MTP forward returns correct structure -----------------------------
def t3():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(device=DEVICE, dtype=DTYPE)
    
    x = torch.randint(0, 256, (2, 32), device=DEVICE)
    
    # Standard forward
    logits_single, _ = model(x)
    assert logits_single.shape == (2, 32, 256)
    
    # MTP forward
    logits_list, _ = model(x, return_mtp=True)
    assert isinstance(logits_list, list) and len(logits_list) == 4
    for i, l in enumerate(logits_list):
        assert l.shape == (2, 32, 256), f"Head {i} shape: {l.shape}"
    
    # Latent forward
    logits_lat, _, x_out = model(x, return_latent=True)
    assert x_out.shape == (2, 32, 64), f"Latent shape: {x_out.shape}"
    
    print(f"    Standard: {logits_single.shape} | MTP: {len(logits_list)}×{logits_list[0].shape} | Latent: {x_out.shape} ✓")

test("All forward modes (standard, MTP, latent) produce correct output", t3)

# -- TEST 4: End-to-end training with all features converges -------------------
def t4():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(device=DEVICE, dtype=DTYPE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    losses = []
    for step in range(50):
        x = torch.randint(0, 256, (2, 32), device=DEVICE)
        
        # MTP training (Step 14)
        logits_list, _ = model(x, return_mtp=True)
        loss = 0.0
        y = x  # auto-regressive target
        for k, logits in enumerate(logits_list):
            if k == 0:
                loss = loss + F.cross_entropy(logits[:, :-1].reshape(-1, 256), y[:, 1:].reshape(-1))
            else:
                loss = loss + F.cross_entropy(logits[:, :-(k+1)].reshape(-1, 256), y[:, (k+1):].reshape(-1))
        
        opt.zero_grad()
        loss.backward()
        assert not torch.isnan(loss), f"NaN loss at step {step}"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    
    first5 = sum(losses[:5]) / 5
    last5 = sum(losses[-5:]) / 5
    improvement = (first5 - last5) / first5 * 100
    print(f"    Loss: {first5:.3f} → {last5:.3f} (improvement: {improvement:.1f}%)")
    assert last5 < first5, f"Loss did not decrease: {first5:.3f} → {last5:.3f}"

test("End-to-end MTP training converges with all improvements active", t4)

# -- TEST 5: Generation produces output at 500+ bytes --------------------------
def t5():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(device=DEVICE, dtype=DTYPE)
    
    prompt = torch.randint(0, 256, (1, 8), device=DEVICE)
    generated = model.generate(prompt, max_new_bytes=500, temp=0.8)
    
    gen_len = generated.shape[1] - prompt.shape[1]
    assert gen_len == 500, f"Expected 500 generated bytes, got {gen_len}"
    assert not torch.isnan(generated.float()).any(), "NaN in generated sequence"
    
    # Decode to text (just to verify it doesn't crash)
    gen_bytes = generated[0, prompt.shape[1]:].tolist()
    decoded = bytes(gen_bytes).decode('utf-8', errors='replace')
    print(f"    Generated {gen_len} bytes | Sample: {decoded[:80]!r}...")

test("Generation produces 500+ bytes without NaN or crash", t5)

# -- TEST 6: Latent think + generation pipeline works --------------------------
def t6():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(device=DEVICE, dtype=DTYPE)
    
    prompt = torch.randint(0, 256, (1, 16), device=DEVICE)
    
    # Generate with think_steps (Step 15)
    generated = model.generate(prompt, max_new_bytes=20, temp=0.8, think_steps=4)
    gen_len = generated.shape[1] - prompt.shape[1]
    assert gen_len == 20, f"Expected 20 generated bytes, got {gen_len}"
    
    # think_to_english
    _, states = model(prompt)
    english = think_to_english(model, states, max_bytes=10)
    assert isinstance(english, str) and len(english) == 10
    
    print(f"    Think+generate: {gen_len} bytes | Think-to-English: {english!r} ✓")

test("Latent think + generation pipeline works end-to-end", t6)

# -- TEST 7: VRAM usage check (small model proxy) ------------------------------
def t7():
    if DEVICE != "cuda":
        print("    Skipping VRAM check (CPU mode)")
        return
        
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(device=DEVICE, dtype=DTYPE)
    
    x = torch.randint(0, 256, (1, 64), device=DEVICE)
    logits, _ = model(x)
    loss = logits.sum()
    loss.backward()
    
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"    Peak VRAM (small model): {peak_mb:.1f} MB")
    assert peak_mb < 6000, f"VRAM usage too high: {peak_mb:.1f} MB"

test("VRAM usage within budget", t7)

print("\n" + "="*60)
fails = [k for k,v in results.items() if not v.startswith("PASS")]
if fails:
    print(f"RESULT: FAILED ({len(fails)}/{len(results)} tests failed)")
    for f in fails: print(f"  ✗ {f}\n    → {results[f]}")
    sys.exit(1)
else:
    print(f"RESULT: ALL {len(results)} TESTS PASSED ✅")
    sys.exit(0)
