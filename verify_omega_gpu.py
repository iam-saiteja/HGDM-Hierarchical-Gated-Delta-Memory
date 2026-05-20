import torch
import torch.nn as nn
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
                step_input = inputs[:, t : t + 1] # shape [B, 1]
                out_step, states_seq = model(step_input, states=states_seq, offset=t)
                out_seq_list.append(out_step)
        out_sequential = torch.cat(out_seq_list, dim=1)
        print(f"Sequential output shape: {out_sequential.shape}")
    except Exception as e:
        print(f"[FAIL] Error during sequential forward pass on GPU: {e}")
        return
        
    max_diff = (out_parallel - out_sequential).abs().max().item()
    print(f"Max absolute difference on GPU: {max_diff:.2e}")
    
    if max_diff < 1e-4:
        print("[SUCCESS] Parallel and sequential outputs are mathematically equivalent on GPU using Triton!")
    else:
        print("[FAIL] GPU output mismatch detected!")

if __name__ == "__main__":
    verify_omega_gpu()
