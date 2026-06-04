"""
HGDM Step 15: HGDM-Think (COCONUT Latent Reasoning) Tests
Run: python3 tests/test_15_hgdm_think.py
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

def clone_states(obj):
    """Recursively clone a nested state structure of tensors, lists, tuples, dicts, and Nones."""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, tuple):
        return tuple(clone_states(x) for x in obj)
    if isinstance(obj, list):
        return [clone_states(x) for x in obj]
    if isinstance(obj, dict):
        return {k: clone_states(v) for k, v in obj.items()}
    return obj  # scalars, strings, etc.

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

print("\n[Step 15] HGDM-Think (COCONUT) Tests")
print("-" * 50)

# -- TEST 1: latent_think runs N steps without emitting tokens and evolves states -----
def t1():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(DEVICE)
    
    # Feed initial prompt to build up states
    prompt = torch.randint(0, 256, (1, 16), device=DEVICE)
    _, states = model(prompt)
    
    # Snapshot states before thinking using detach+clone
    states_before = clone_states(states)
    
    # Run latent thinking with enough steps to trigger at least one boundary
    # (sigmoid(-2.08) ≈ 0.111 per step; need ~10 steps for cum_prob >= 1.0)
    n_thoughts = 16
    next_states, thought_tokens = latent_think(model, states, n_thoughts=n_thoughts)
    
    assert len(thought_tokens) == n_thoughts, f"Expected {n_thoughts} thought tokens, got {len(thought_tokens)}"
    
    # Renderer states (states[2]) evolve on EVERY step regardless of boundary trigger.
    # Core states (states[1]) evolve only when boundary fires (~every 9 steps).
    # Check renderer state evolution first (guaranteed to change).
    ren_S_before = states_before[2][0][0]  # renderer layer 0, S matrix
    ren_S_after = next_states[2][0][0]
    ren_diff = (ren_S_before - ren_S_after).abs().sum().item()
    print(f"    Renderer state diff: {ren_diff:.6f}")
    assert ren_diff > 1e-4, f"Renderer states did not evolve: diff={ren_diff}"
    
    # Check core state evolution (should trigger at least once in 16 steps)
    core_S_before = states_before[1][0][0]  # core layer 0, S matrix
    core_S_after = next_states[1][0][0]
    core_diff = (core_S_before - core_S_after).abs().sum().item()
    print(f"    Core state diff: {core_diff:.6f}")
    assert core_diff > 1e-4, f"Core states did not evolve: diff={core_diff}"
    
test("latent_think() runs N steps and evolves states without emitting tokens", t1)

# -- TEST 2: Generated text after thinking is different from without thinking ----------
def t2():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(device=DEVICE, dtype=DTYPE)
    
    # Assign random weights to make predictions interesting
    for p in model.parameters():
        if p.requires_grad:
            p.data.normal_(std=0.1)
            
    prompt = torch.randint(0, 256, (1, 16), device=DEVICE)
    
    # Generate without thinking
    gen_no_think = model.generate(prompt, max_new_bytes=10, temp=0.8, think_steps=0)
    
    # Generate with thinking (5 steps of think before each byte)
    gen_with_think = model.generate(prompt, max_new_bytes=10, temp=0.8, think_steps=5)
    
    no_think_list = gen_no_think[0].tolist()
    with_think_list = gen_with_think[0].tolist()
    
    print(f"    No-think output (last 10): {no_think_list[-10:]}")
    print(f"    With-think output (last 10): {with_think_list[-10:]}")
    
    assert no_think_list != with_think_list, "Generated sequence is identical with and without thinking"
    
test("Generated text after thinking is different from without thinking", t2)

# -- TEST 3: No NaN in any think step -------------------------------------------
def t3():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(device=DEVICE, dtype=DTYPE)
    
    prompt = torch.randint(0, 256, (1, 16), device=DEVICE)
    _, states = model(prompt)
    
    # Run a long latent think sequence (100 steps = ~11 boundary triggers)
    next_states, thought_tokens = latent_think(model, states, n_thoughts=100)
    
    # Check for NaNs in core states
    for i, layer_state in enumerate(next_states[1]):
        S, n = layer_state
        assert not torch.isnan(S).any(), f"NaN found in core layer {i} state S"
        assert not torch.isnan(n).any(), f"NaN found in core layer {i} state n"
        
    # Check for NaNs in renderer states
    for i, layer_state in enumerate(next_states[2]):
        S, n = layer_state
        assert not torch.isnan(S).any(), f"NaN found in renderer layer {i} state S"
        assert not torch.isnan(n).any(), f"NaN found in renderer layer {i} state n"

    print("    No NaN values detected in core or renderer states after 100 think steps ✓")

test("No NaN values in states after long think trajectory", t3)

# -- TEST 4: think_to_english() produces readable ASCII output -------------------
def t4():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(device=DEVICE, dtype=DTYPE)
    
    prompt = torch.randint(0, 256, (1, 16), device=DEVICE)
    _, states = model(prompt)
    
    english_thoughts = think_to_english(model, states, max_bytes=30)
    print(f"    Latent thoughts translation: {english_thoughts!r}")
    
    assert isinstance(english_thoughts, str), "Expected string output"
    assert len(english_thoughts) == 30, f"Expected string length of 30, got {len(english_thoughts)}"
    
    # Every character must be printable ASCII, newline, or the '.' fallback
    for char in english_thoughts:
        code = ord(char)
        assert (32 <= code <= 126) or char in ['\n', '\r'], f"Non-printable char found: {char!r} (ord={code})"

test("think_to_english() produces readable ASCII output of correct length", t4)

print("\n" + "="*60)
fails = [k for k,v in results.items() if not v.startswith("PASS")]
if fails:
    print(f"RESULT: FAILED ({len(fails)}/{len(results)} tests failed)")
    for f in fails: print(f"  ✗ {f}\n    → {results[f]}")
    sys.exit(1)
else:
    print(f"RESULT: ALL {len(results)} TESTS PASSED ✅")
    sys.exit(0)
