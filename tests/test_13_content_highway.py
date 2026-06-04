"""
HGDM Step 13: Input-Dependent Highway Gate Tests
Run: python3 tests/test_13_content_highway.py
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

print("\n[Step 13] Input-Dependent Highway Gate Tests")
print("-" * 50)

# -- TEST 1: Gate networks exists and shapes check ------------------------------
def t1():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(DEVICE)
    
    assert hasattr(model, 'td_gate_net'), "td_gate_net not found"
    assert hasattr(model, 'bu_gate_net'), "bu_gate_net not found"
    
    # td_gate_net: d_model -> H_ren (4)
    assert model.td_gate_net.in_features == 256
    assert model.td_gate_net.out_features == 4
    
    # bu_gate_net: d_byte -> H_core (4)
    assert model.bu_gate_net.in_features == 64
    assert model.bu_gate_net.out_features == 4
    
    # Initialized to produce near-zero gates
    td_bias = model.td_gate_net.bias.data
    bu_bias = model.bu_gate_net.bias.data
    
    assert torch.allclose(td_bias, torch.full_like(td_bias, -4.0)), f"Expected td bias -4.0, got {td_bias}"
    assert torch.allclose(bu_bias, torch.full_like(bu_bias, -2.0)), f"Expected bu bias -2.0, got {bu_bias}"
    assert torch.allclose(model.td_gate_net.weight, torch.zeros_like(model.td_gate_net.weight)), "Expected td weight = 0"
    assert torch.allclose(model.bu_gate_net.weight, torch.zeros_like(model.bu_gate_net.weight)), "Expected bu weight = 0"
    
test("Gate networks exist and are correctly initialized to near-zero", t1)

# -- TEST 2: Highway gates respond dynamically to different inputs in batch mode ----
def t2():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(device=DEVICE, dtype=DTYPE)
    
    # Assign small random weights to prevent sigmoid saturation in bfloat16
    with torch.no_grad():
        model.td_gate_net.weight.data.normal_(std=0.05)
        model.bu_gate_net.weight.data.normal_(std=0.05)
        
    td_gates = []
    bu_gates = []
    
    def td_hook(module, input, output):
        td_gates.append(torch.sigmoid(output).clone())
    def bu_hook(module, input, output):
        bu_gates.append(torch.sigmoid(output).clone())
        
    h1 = model.td_gate_net.register_forward_hook(td_hook)
    h2 = model.bu_gate_net.register_forward_hook(bu_hook)
    
    # Run two different inputs
    x1 = torch.randint(0, 256, (2, 32), device=DEVICE)
    x2 = torch.randint(0, 256, (2, 32), device=DEVICE)
    # Ensure they are different
    while torch.equal(x1, x2):
        x2 = torch.randint(0, 256, (2, 32), device=DEVICE)
        
    # We must have prior states to trigger bu_highway in the batch path
    states = [
        [None]*config.catcher_layers, 
        [None]*config.core_layers, 
        [None]*config.renderer_layers, 
        None, 
        {
            'prev_renderer_last_S': (
                torch.randn(2, 4, 32, 32, device=DEVICE, dtype=DTYPE), 
                torch.randn(2, 4, 32, device=DEVICE, dtype=DTYPE)
            ),
            'x_dec_last': torch.randn(2, 64, device=DEVICE, dtype=DTYPE)
        }
    ]
    
    import copy
    states1 = copy.deepcopy(states)
    model(x1, states=states1)
    
    states2 = copy.deepcopy(states)
    # Change the x_dec_last to be different
    states2[4]['x_dec_last'] = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)
    model(x2, states=states2)
    
    h1.remove()
    h2.remove()
    
    assert len(td_gates) == 2, f"Expected 2 activations of td_gate_net, got {len(td_gates)}"
    assert len(bu_gates) == 2, f"Expected 2 activations of bu_gate_net, got {len(bu_gates)}"
    
    # Verify gates are in [0, 1] range and not saturated
    for g in td_gates + bu_gates:
        assert torch.all(g >= 0.0) and torch.all(g <= 1.0), "Gate values not in [0, 1]"
        assert not torch.all(g == 0.0) and not torch.all(g == 1.0), "Gate values are fully saturated"
        
    # Verify the gate values respond to inputs and are different
    td_diff = (td_gates[0] - td_gates[1]).abs().sum().item()
    bu_diff = (bu_gates[0] - bu_gates[1]).abs().sum().item()
    
    print(f"    td gate diff: {td_diff:.6f} | bu gate diff: {bu_diff:.6f}")
    assert td_diff > 1e-4, "td_gate did not respond to input differences"
    assert bu_diff > 1e-4, "bu_gate did not respond to input differences"

test("Highway gates respond dynamically in batch mode and are in (0,1)", t2)

# -- TEST 3: Highway gates respond dynamically to different inputs in step-by-step mode ----
def t3():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(device=DEVICE, dtype=DTYPE)
    
    # Assign small random weights to prevent sigmoid saturation in bfloat16
    with torch.no_grad():
        model.td_gate_net.weight.data.normal_(std=0.05)
        model.bu_gate_net.weight.data.normal_(std=0.05)
        
    # Force boundary probability to trigger on every step
    with torch.no_grad():
        model.boundary_head.bias.fill_(10.0)
        
    td_gates = []
    bu_gates = []
    
    def td_hook(module, input, output):
        td_gates.append(torch.sigmoid(output).clone())
    def bu_hook(module, input, output):
        bu_gates.append(torch.sigmoid(output).clone())
        
    h1 = model.td_gate_net.register_forward_hook(td_hook)
    h2 = model.bu_gate_net.register_forward_hook(bu_hook)
    
    # Run step mode (T=1) on different inputs
    # Need to setup states with non-None prev_renderer_last_S and initial cum_prob=0.5
    # to guarantee trigger_flag triggers on first step.
    states = [
        [None]*config.catcher_layers, 
        [None]*config.core_layers, 
        [None]*config.renderer_layers, 
        None, 
        {
            'cum_prob': torch.full((2,), 0.5, device=DEVICE, dtype=DTYPE),
            'z_broadcast_cache': torch.zeros(2, 64, device=DEVICE, dtype=DTYPE),
            'prev_renderer_last_S': (
                torch.randn(2, 4, 32, 32, device=DEVICE, dtype=DTYPE), 
                torch.randn(2, 4, 32, device=DEVICE, dtype=DTYPE)
            )
        }
    ]
    
    import copy
    states1 = copy.deepcopy(states)
    x1 = torch.randint(0, 256, (2, 1), device=DEVICE)
    _, next_states1 = model(x1, states=states1)
    
    states2 = copy.deepcopy(states)
    x2 = torch.randint(0, 256, (2, 1), device=DEVICE)
    # Ensure they are different
    while torch.equal(x1, x2):
        x2 = torch.randint(0, 256, (2, 1), device=DEVICE)
    _, next_states2 = model(x2, states=states2)
    
    h1.remove()
    h2.remove()
    
    assert len(td_gates) == 2, f"Expected 2 activations of td_gate_net in step mode, got {len(td_gates)}"
    assert len(bu_gates) == 2, f"Expected 2 activations of bu_gate_net in step mode, got {len(bu_gates)}"
    
    # Verify gates are in [0, 1] range and not saturated
    for g in td_gates + bu_gates:
        assert torch.all(g >= 0.0) and torch.all(g <= 1.0), "Step-mode gate values not in [0, 1]"
        assert not torch.all(g == 0.0) and not torch.all(g == 1.0), "Step-mode gate values are fully saturated"
        
    # Verify they are different
    td_diff = (td_gates[0] - td_gates[1]).abs().sum().item()
    bu_diff = (bu_gates[0] - bu_gates[1]).abs().sum().item()
    
    print(f"    Step mode td gate diff: {td_diff:.6f} | bu gate diff: {bu_diff:.6f}")
    assert td_diff > 1e-4, "Step-mode td_gate did not respond to input differences"
    assert bu_diff > 1e-4, "Step-mode bu_gate did not respond to input differences"

test("Highway gates respond dynamically in step-by-step mode and are in (0,1)", t3)

# -- TEST 4: Gradient flow through gate networks ---------------------------------
def t4():
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512
    )
    model = OmegaGDM(config).to(device=DEVICE, dtype=DTYPE)
    
    x = torch.randint(0, 256, (2, 32), device=DEVICE)
    # We must have prior states with random values to trigger bu_highway and allow gradient flow
    states = [
        [None]*config.catcher_layers, 
        [None]*config.core_layers, 
        [None]*config.renderer_layers, 
        None, 
        {
            'prev_renderer_last_S': (
                torch.randn(2, 4, 32, 32, device=DEVICE, dtype=DTYPE), 
                torch.randn(2, 4, 32, device=DEVICE, dtype=DTYPE)
            ),
            'x_dec_last': torch.randn(2, 64, device=DEVICE, dtype=DTYPE)
        }
    ]
    
    logits, _ = model(x, states=states)
    loss = logits.sum()
    loss.backward()
    
    td_grad_norm = model.td_gate_net.weight.grad.norm().item()
    bu_grad_norm = model.bu_gate_net.weight.grad.norm().item()
    
    print(f"    td_gate_net grad norm: {td_grad_norm:.6f} | bu_gate_net grad norm: {bu_grad_norm:.6f}")
    assert td_grad_norm > 0.0, "td_gate_net weight gradient is zero"
    assert bu_grad_norm > 0.0, "bu_gate_net weight gradient is zero"

test("Gradient: nonzero gradients flow to td_gate_net and bu_gate_net", t4)

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
        states = [
            [None]*config.catcher_layers, 
            [None]*config.core_layers, 
            [None]*config.renderer_layers, 
            None, 
            {
                'prev_renderer_last_S': (
                    torch.randn(2, 4, 32, 32, device=DEVICE, dtype=DTYPE), 
                    torch.randn(2, 4, 32, device=DEVICE, dtype=DTYPE)
                ),
                'x_dec_last': torch.randn(2, 64, device=DEVICE, dtype=DTYPE)
            }
        ]
        logits, _ = model(x, states=states)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1))
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

test("Training: loss decreases and converges without NaNs", t5)

print("\n" + "="*60)
fails = [k for k,v in results.items() if not v.startswith("PASS")]
if fails:
    print(f"RESULT: FAILED ({len(fails)}/{len(results)} tests failed)")
    for f in fails: print(f"  ✗ {f}\n    → {results[f]}")
    sys.exit(1)
else:
    print(f"RESULT: ALL {len(results)} TESTS PASSED ✅")
    sys.exit(0)
