import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import time
import json
from hgdm_ultimate import HGDMUltimate, HGDMConfig
from utils import get_gpu_memory_usage

def measure_kernel_impact():
    device = torch.device('cuda')
    lengths = [512, 2048, 4096]
    
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    
    results = {
        "lengths": lengths,
        "fused": {"speed": [], "vram": []},
        "sequential": {"speed": [], "vram": []}
    }
    
    print(f"\n{'='*50}\nExp 8: Kernel Impact (Fused vs Sequential)\n{'='*50}")
    
    for mode in ["fused", "sequential"]:
        print(f"\n--- Testing Mode: {mode.upper()} ---")
        force_seq = (mode == "sequential")
        model = HGDMUltimate(config, force_sequential=force_seq).to(device)
        model.eval()
        
        for L in lengths:
            x = torch.randint(0, 256, (1, L), device=device)
            
            # Warmup
            torch.cuda.empty_cache()
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    for _ in range(3): _ = model(x)
            
            # Benchmark
            t0 = time.time()
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    for _ in range(10): _ = model(x)
            t_total = time.time() - t0
            avg_time = t_total / 10
            speed = L / avg_time
            vram = get_gpu_memory_usage()
            
            results[mode]["speed"].append(speed)
            results[mode]["vram"].append(vram)
            print(f"L={L:5d} | {speed:8.0f} tok/s | VRAM: {vram:5.0f}MB")
            
        del model
        torch.cuda.empty_cache()
        
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nExperiment 8 Complete. Saved results.json")

if __name__ == "__main__":
    measure_kernel_impact()
