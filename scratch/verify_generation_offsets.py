import torch
import torch.nn as nn
import os
import sys

# Ensure HTSPC workspace root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hgdm_ultimate import HGDMUltimate, HGDMConfig

def run_test():
    print("="*60)
    print("STEP 1 VERIFICATION: GENERATION OFFSET TRACKING")
    print("="*60)

    # 1. Initialize a small 2-layer model
    config = HGDMConfig(
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_k=16,
        d_v=16,
        d_ff=128,
        vocab_size=256
    )
    # Force sequential path to avoid Triton dependency during simple verification
    model = HGDMUltimate(config, force_sequential=True)
    model.eval()

    # 2. Zero the token embedding weight so x starts as 0 before pos_embedding is added
    model.embedding.weight.data.zero_()

    # 3. Setup pos_embedding to be a recognizable escalating value at each position index
    # We set coordinate [0, t, d] = float(t)
    pos_escalation = torch.arange(65536, dtype=torch.float32).view(1, 65536, 1).expand(1, 65536, config.d_model).clone()
    model.pos_embedding.data = pos_escalation

    # 4. Register a forward pre-hook on the first layer's norm to capture the exactly added positional embedding
    captured_embeddings = []
    def norm_hook(module, args):
        # args[0] is the input to the module, shape (B, T, d_model)
        # We average across the last dimension to get the scalar coordinate value
        input_x = args[0]
        captured_embeddings.append(input_x[0, :, 0].clone())

    # Register the hook
    hook_handle = model.layers[0].norm1.register_forward_pre_hook(norm_hook)

    # 5. Run generation
    prompt = torch.tensor([[10, 20, 30]], dtype=torch.long) # Length 3
    print(f"Running generation with prompt length {prompt.shape[1]}...")
    _ = model.generate(prompt, max_new_bytes=5) # Should generate 5 new tokens

    hook_handle.remove()

    # 6. Analyze captured positional embeddings
    # The first forward call (prompt):
    # Length of prompt is 3. So offset = 0, T = 3.
    # The added pos_embedding should be [0.0, 1.0, 2.0].
    #
    # The subsequent autoregressive generation steps (5 steps):
    # Step 1: next_byte at offset 3. Should add [3.0].
    # Step 2: next_byte at offset 4. Should add [4.0].
    # Step 3: next_byte at offset 5. Should add [5.0].
    # Step 4: next_byte at offset 6. Should add [6.0].
    # Step 5: next_byte at offset 7. Should add [7.0].
    
    print("\nCaptured positional embedding scalars at each forward step:")
    for idx, cap in enumerate(captured_embeddings):
        vals = cap.tolist()
        print(f"  Forward Call {idx:2d} | Shape: {cap.shape} | Vals: {vals}")

    # Assert correctness
    assert len(captured_embeddings) == 5, f"Expected 5 forward calls (1 prompt + 4 auto-regressive), got {len(captured_embeddings)}"
    
    # Prompt verification
    assert torch.allclose(captured_embeddings[0], torch.tensor([0.0, 1.0, 2.0])), "Prompt positional embeddings are incorrect!"
    
    # Auto-regressive step verifications (max_new_bytes - 1 = 4 steps)
    expected_steps = [3.0, 4.0, 5.0, 6.0]
    for i, exp_val in enumerate(expected_steps):
        actual_val = captured_embeddings[i + 1].item()
        assert abs(actual_val - exp_val) < 1e-5, f"Step {i} expected positional offset {exp_val}, but got {actual_val}"

    print("\n[SUCCESS] Positional embedding offset tracking works perfectly!")
    print("Each newly generated token receives the correct escalating offset index.")
    print("="*60)

if __name__ == "__main__":
    run_test()
