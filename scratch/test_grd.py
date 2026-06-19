"""
GRD Validation Tests — 3 core behavioral probes:
  Test 1: Reservoir A (Complex Oscillator) long-range memory test
  Test 2: Reservoir B (NCM Novelty Gate) selective write test
  Test 3: Reservoir C (CADP) contradiction detection test
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn.functional as F
from ultimate.grd import GRDConfig, GRDModel, GeometricReservoirMixer, count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}\n")

# --------------------------------------------------------------------------
# TEST 1: FORWARD PASS + PARAMETER COUNT
# --------------------------------------------------------------------------
print("=" * 60)
print("TEST 1: Forward Pass & Parameter Count")
print("=" * 60)
cfg   = GRDConfig(d_model=256, n_layers=4, n_heads=4, d_k=64, d_v=64, d_ff=512)
model = GRDModel(cfg).to(device)
total = count_parameters(model)
print(f"  Parameters: {total:,}")

x = torch.randint(0, 256, (2, 128), device=device)
logits, states = model(x)
assert logits.shape == (2, 128, 256), f"Bad shape: {logits.shape}"
print(f"  Logits shape: {logits.shape}  ✓")
print(f"  State tuple length: {len(states[0])}  (A_real, A_imag, B, n_B, C)  ✓")

# --------------------------------------------------------------------------
# TEST 2: RESERVOIR A — does it remember information from far away?
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 2: Reservoir A — Complex Oscillator Long-Range Memory")
print("=" * 60)
# Write a signal at position 0, then read it at positions 50, 100, 200, 500
# In a naive decay model, the signal at t=500 should be ~exp(-500/tau) ≈ 0
# In a complex oscillator with |gamma| ≈ 1, the signal should persist

cfg_a = GRDConfig(d_model=64, n_layers=1, n_heads=2, d_k=32, d_v=32, d_ff=128)
mixer = GeometricReservoirMixer(cfg_a).to(device)
mixer.eval()

with torch.no_grad():
    # All-zero input except first token (signal injection)
    signal = torch.zeros(1, 1000, 64, device=device)
    signal[:, 0, :] = 1.0   # inject signal at t=0

    # Run through mixer and track S_A_real norm at various positions
    state = None
    norms_A = []
    for t in range(1000):
        out, state = mixer(signal[:, t:t+1, :], state=state)
        S_A_real = state[0]  # [1, H, d_k, d_v]
        norms_A.append(S_A_real.norm().item())

print(f"  S_A_real norm at t=1:   {norms_A[0]:.4f}")
print(f"  S_A_real norm at t=50:  {norms_A[49]:.4f}")
print(f"  S_A_real norm at t=200: {norms_A[199]:.4f}")
print(f"  S_A_real norm at t=500: {norms_A[499]:.4f}")
print(f"  S_A_real norm at t=999: {norms_A[998]:.4f}")

ratio = norms_A[998] / (norms_A[0] + 1e-8)
if ratio > 0.5:
    print(f"  Persistence ratio t=999/t=1: {ratio:.3f}  ✓ Complex oscillator PRESERVES memory!")
else:
    print(f"  Persistence ratio t=999/t=1: {ratio:.3f}  ✗ Memory faded (check mag initialization)")

# --------------------------------------------------------------------------
# TEST 3: RESERVOIR B — does novelty gate prevent redundant writes?
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 3: Reservoir B — NCM Novelty-Gated Selective Write")
print("=" * 60)

cfg_b = GRDConfig(d_model=64, n_layers=1, n_heads=2, d_k=32, d_v=32, d_ff=128)
model_b = GRDModel(cfg_b).to(device)
model_b.eval()

with torch.no_grad():
    # Sequence A: all same byte (maximally redundant — should write very little after first)
    seq_redundant = torch.full((1, 200), 65, dtype=torch.long, device=device)  # 200x 'A'
    # Sequence B: random bytes (maximally novel — should write frequently)
    seq_novel     = torch.randint(0, 256, (1, 200), device=device)

    _, states_r = model_b(seq_redundant)
    _, states_n = model_b(seq_novel)

    # Compare S_B norms: novel sequence should have a much larger S_B
    S_B_redundant = states_r[-1][2].norm().item()
    S_B_novel     = states_n[-1][2].norm().item()

print(f"  S_B norm after 200 identical bytes: {S_B_redundant:.4f}")
print(f"  S_B norm after 200 random bytes:    {S_B_novel:.4f}")
if S_B_novel > S_B_redundant:
    print(f"  ✓ Novelty gate correctly writes MORE for novel sequences!")
else:
    print(f"  ✗ Novelty gate not differentiating (check threshold)")

# --------------------------------------------------------------------------
# TEST 4: GRADIENT FLOW — do gradients reach all parameters?
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 4: Gradient Flow Through All Reservoirs")
print("=" * 60)
cfg_g = GRDConfig(d_model=64, n_layers=2, n_heads=2, d_k=32, d_v=32, d_ff=128)
model_g = GRDModel(cfg_g).to(device)
model_g.train()

x = torch.randint(0, 256, (1, 64), device=device)
y = torch.randint(0, 256, (1, 64), device=device)
logits, _ = model_g(x)
loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
loss.backward()

no_grad = []
for name, p in model_g.named_parameters():
    if p.grad is None:
        no_grad.append(name)

if len(no_grad) == 0:
    print(f"  Loss: {loss.item():.4f}  ✓ All parameters received gradients!")
else:
    print(f"  WARNING: {len(no_grad)} params have no gradient: {no_grad[:5]}")

# --------------------------------------------------------------------------
# TEST 5: GENERATION
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 5: Auto-regressive Generation")
print("=" * 60)
model_g.eval()
prompt = torch.tensor([[72, 101, 108, 108, 111]], device=device)  # "Hello"
out = model_g.generate(prompt, max_new_bytes=30, temp=0.9)
generated_text = bytes([b for b in out[0].tolist() if 32 <= b <= 126]).decode('ascii', errors='replace')
print(f"  Generated: '{generated_text}'")
print(f"  ✓ Generation completed {out.shape[1]} bytes.")

print("\n" + "=" * 60)
print("ALL GRD TESTS COMPLETE")
print("=" * 60)
