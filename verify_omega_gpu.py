import torch
import torch.nn as nn

# Disable TF32 on Ampere GPUs to verify exact math equivalence without hardware truncation
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from hgdm_omega import OmegaConfig, OmegaGDM

def verify_omega_gpu():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA is not available. This test must run on the GPU server.")
        return
        
    device = torch.device("cuda")
    print(f"================ Running GPU Verification ================")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Configure a representative OmegaGDM model to run on GPU
    config = OmegaConfig(
        d_byte=256,
        catcher_layers=2,
        renderer_layers=2,
        d_model=1024,
        core_layers=4,
        n_heads=8,
        d_k=64,
        d_v=64,
        d_ff=4096,
        decimation_rate=8, # W = 8
        vocab_size=256
    )
    
    # force_sequential=False enables the Triton fused_nitro_scan kernel
    model = OmegaGDM(config, force_sequential=False).to(device)
    model.eval()
    
    B = 2
    T = 64
    inputs = torch.randint(0, 256, (B, T), device=device)
    
    # -------------------------------------------------------------------------
    # TEST 1: Pure PyTorch Sequential Path (Force math equivalence check)
    # -------------------------------------------------------------------------
    print("\n--- TEST 1: Pure PyTorch Path (force_sequential=True) ---")
    model_py = OmegaGDM(config, force_sequential=True).to(device)
    # Copy weights to ensure identical initialization
    model_py.load_state_dict(model.state_dict())
    model_py.eval()
    
    with torch.no_grad():
        out_parallel_py, _ = model_py(inputs)
        
        states_seq_py = None
        out_seq_list_py = []
        for t in range(T):
            step_input = inputs[:, t : t + 1]
            out_step, states_seq_py = model_py(step_input, states=states_seq_py, offset=t)
            out_seq_list_py.append(out_step)
        out_sequential_py = torch.cat(out_seq_list_py, dim=1)
        
    diff_py = (out_parallel_py - out_sequential_py).abs().max().item()
    print(f"Max absolute difference (Pure PyTorch): {diff_py:.2e}")
    if diff_py < 1e-5:
        print("[SUCCESS] Pure PyTorch parallel & sequential paths are mathematically identical!")
    else:
        print("[FAIL] Pure PyTorch path mismatch!")

    # -------------------------------------------------------------------------
    # TEST 2: Triton Fused Path (force_sequential=False)
    # -------------------------------------------------------------------------
    print("\n--- TEST 2: Triton Fused Path (force_sequential=False) ---")
    print("Running parallel path on GPU (triggering Triton JIT)...")
    try:
        with torch.no_grad():
            out_parallel, states_parallel = model(inputs)
        print(f"Parallel output shape: {out_parallel.shape}")
    except Exception as e:
        print(f"[FAIL] Error during parallel forward pass on GPU: {e}")
        return
        
    print("Running sequential path step-by-step on GPU...")
    states_seq = None
    out_seq_list = []
    
    try:
        with torch.no_grad():
            for t in range(T):
                step_input = inputs[:, t : t + 1]
                out_step, states_seq = model(step_input, states=states_seq, offset=t)
                out_seq_list.append(out_step)
        out_sequential = torch.cat(out_seq_list, dim=1)
        print(f"Sequential output shape: {out_sequential.shape}")
    except Exception as e:
        print(f"[FAIL] Error during sequential forward pass on GPU: {e}")
        return
        
    max_diff = (out_parallel - out_sequential).abs().max().item()
    print(f"Max absolute difference (Triton): {max_diff:.2e}")
    
    if max_diff < 1e-3:
        print("[SUCCESS] Triton parallel & sequential paths are numerically consistent within float32 tolerance!")
    else:
        print("[FAIL] Triton path numerical mismatch exceeds tolerance!")

if __name__ == "__main__":
    verify_omega_gpu()
