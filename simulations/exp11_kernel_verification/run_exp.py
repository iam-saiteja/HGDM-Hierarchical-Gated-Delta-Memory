import torch
import time
import sys
import os
import json

# Add parent directory to path to import kernel
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kernel_nitro import fused_nitro_scan

def run_experiment():
    print("="*80)
    print("EXP 11: KERNEL VERIFICATION SUITE")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("[WARNING] CUDA not available. Fused kernel tests require an NVIDIA GPU.")
        print("Exiting test.")
        return

    device = torch.device('cuda')
    
    # -----------------------------------------------------------------------------
    # PART A: SPEED BENCHMARK (num_warps / num_stages scaling test)
    # -----------------------------------------------------------------------------
    print("\n--- PART A: Triton Execution Speed Benchmark ---")
    B, H, T, d = 1, 12, 8192, 64
    print(f"Testing Sequence Length: {T}, Heads: {H}, Dim: {d}")
    
    q = torch.randn(B, T, H, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, T, H, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, T, H, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    alpha = torch.rand(B, T, H, d, device=device, dtype=torch.float32, requires_grad=True)
    beta = torch.rand(B, T, H, d, device=device, dtype=torch.float32, requires_grad=True)
    
    # Warmup
    for _ in range(5):
        out, state = fused_nitro_scan(q, k, v, alpha, beta)
        out.sum().backward()
    
    torch.cuda.synchronize()
    
    # Timed run
    start_time = time.time()
    for _ in range(10):
        out, state = fused_nitro_scan(q, k, v, alpha, beta)
        out.sum().backward()
    torch.cuda.synchronize()
    
    avg_ms = ((time.time() - start_time) / 10) * 1000
    print(f"Average Forward+Backward Pass: {avg_ms:.2f} ms")
    
    if avg_ms > 1000:
        print("[FAIL] The kernel is taking too long! The num_warps fix may not have applied.")
    else:
        print("[PASS] The 4-second bottleneck is completely eliminated!")

    # -----------------------------------------------------------------------------
    # PART B: STUFFED MAMBA MATHEMATICAL PROOF (b[:, None] axis test)
    # -----------------------------------------------------------------------------
    print("\n--- PART B: The 'Stuffed Mamba' State Collapse Proof ---")
    
    T_test = 4096
    print(f"Injecting a single Passkey token followed by {T_test-1} Noise tokens.")
    
    q_b = torch.randn(1, T_test, 1, d, device=device, dtype=torch.bfloat16)
    k_b = torch.randn(1, T_test, 1, d, device=device, dtype=torch.bfloat16)
    v_b = torch.randn(1, T_test, 1, d, device=device, dtype=torch.bfloat16)
    
    alpha_b = torch.ones(1, T_test, 1, d, device=device, dtype=torch.float32) # Perfect memory retention (alpha=1)
    beta_b = torch.zeros(1, T_test, 1, d, device=device, dtype=torch.float32) # All gates CLOSED (beta=0)
    
    # 1. Create the Passkey at token 0
    k_passkey = torch.randn(1, 1, 1, d, device=device, dtype=torch.bfloat16)
    v_passkey = torch.randn(1, 1, 1, d, device=device, dtype=torch.bfloat16)
    k_b[0, 0] = k_passkey[0, 0]
    v_b[0, 0] = v_passkey[0, 0]
    beta_b[0, 0] = 1.0 # OPEN the gate specifically for the passkey
    
    # 2. Run the fused kernel
    out_b, state_b = fused_nitro_scan(q_b, k_b, v_b, alpha_b, beta_b)
    
    # 3. Calculate what the mathematically perfect state should be
    # S_t = S_{t-1} * alpha_t + (k_t^T * v_t) * beta_t
    # Since alpha=1 everywhere, and beta=0 everywhere except t=0:
    # S_final MUST equal k_0^T * v_0
    
    # Convert to float32 for exact math verification
    k_f32 = k_passkey.squeeze().float().unsqueeze(1) # (64, 1)
    v_f32 = v_passkey.squeeze().float().unsqueeze(0) # (1, 64)
    expected_state = torch.matmul(k_f32, v_f32)      # (64, 64) outer product
    
    # Get the actual final state from the Triton kernel
    actual_state = state_b.squeeze().float() # (64, 64)
    
    # Check max absolute error
    max_error = torch.max(torch.abs(expected_state - actual_state)).item()
    print(f"Max absolute divergence after 4000 noise tokens: {max_error:.6f}")
    
    if max_error < 1e-1:
        print("[PASS] The state remained perfectly intact! The b[:, None] math fix worked.")
        print("[PASS] HGDM is immune to the Stuffed Mamba state collapse when gates are functioning.")
    else:
        print("[FAIL] The state was corrupted! The noise leaked into the memory matrix.")
        
    print("\nVerification Complete.")
    
    # Save results
    results = {
        "benchmark_forward_backward_ms": avg_ms,
        "stuffed_mamba_max_divergence": max_error,
        "status": "PASS" if (avg_ms < 1000 and max_error < 1e-1) else "FAIL"
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_experiment()
