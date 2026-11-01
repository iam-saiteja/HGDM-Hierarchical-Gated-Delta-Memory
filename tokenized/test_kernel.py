import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from kernel_nitro import fused_nitro_scan_with_n
    print("[Kernel Check] Import fused_nitro_scan_with_n: SUCCESS")
except Exception as e:
    print(f"[Kernel Check] Import FAILED: {e}")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Kernel Check] Device: {device}")

if device == "cuda":
    B, T, H, dk, dv = 2, 64, 6, 64, 64
    q = torch.randn(B, T, H, dk, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, dk, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, dv, device=device, dtype=torch.bfloat16)
    a = torch.rand(B, T, H, device=device, dtype=torch.bfloat16)
    b = torch.rand(B, T, H, device=device, dtype=torch.bfloat16)
    
    try:
        out, S, n = fused_nitro_scan_with_n(q, k, v, a, b, chunk_size=32)
        print(f"[Kernel Check] Forward pass SUCCESS! Out shape: {out.shape}, S shape: {S.shape}")
    except Exception as e:
        print(f"[Kernel Check] Forward pass FAILED: {e}")
