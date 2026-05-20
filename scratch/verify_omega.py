import torch
import torch.nn as nn
from hgdm_omega import OmegaConfig, OmegaGDM

def verify_omega_equivalence():
    print("======================================================")
    # Configure a tiny OmegaGDM model to run locally on CPU
    config = OmegaConfig(
        d_byte=32,
        catcher_layers=1,
        renderer_layers=1,
        d_model=64,
        core_layers=1,
        n_heads=2,
        d_k=32,
        d_v=32,
        d_ff=128,
        decimation_rate=4, # W = 4
        vocab_size=256
    )
    
    # Instantiate model with force_sequential=True to verify on CPU (without Triton)
    model = OmegaGDM(config, force_sequential=True)
    model.eval()
    
    # 1. Create a dummy sequence of length 8 (2 complete blocks of W=4)
    B = 2
    T = 8
    inputs = torch.randint(0, 256, (B, T))
    
    print(f"Inputs: {inputs}")
    print(f"Running parallel path for sequence of length {T}...")
    
    # 2. Run parallel forward pass
    with torch.no_grad():
        out_parallel, states_parallel = model(inputs)
        
    print(f"Parallel output shape: {out_parallel.shape}")
    
    # 3. Run sequential step-by-step forward pass
    print("Running sequential path step-by-step...")
    states_seq = None
    out_seq_list = []
    
    with torch.no_grad():
        for t in range(T):
            step_input = inputs[:, t : t + 1] # shape [B, 1]
            out_step, states_seq = model(step_input, states=states_seq, offset=t)
            out_seq_list.append(out_step)
            
    out_sequential = torch.cat(out_seq_list, dim=1)
    print(f"Sequential output shape: {out_sequential.shape}")
    
    # 4. Check for equivalence
    max_diff = (out_parallel - out_sequential).abs().max().item()
    print(f"Max absolute difference between parallel and sequential outputs: {max_diff:.2e}")
    
    if max_diff < 1e-5:
        print("[SUCCESS] Parallel and sequential outputs are mathematically equivalent!")
    else:
        print("[FAIL] Output mismatch detected!")
        # Let's inspect step-by-step values if they mismatch
        for t in range(T):
            diff_t = (out_parallel[:, t] - out_sequential[:, t]).abs().max().item()
            print(f"  Step {t}: max diff = {diff_t:.2e}")
            
    # Check states equivalence at the end of sequence
    # Let's compare catcher states, core states, renderer states, and buffer states
    print("Checking state structures...")
    
if __name__ == "__main__":
    verify_omega_equivalence()
