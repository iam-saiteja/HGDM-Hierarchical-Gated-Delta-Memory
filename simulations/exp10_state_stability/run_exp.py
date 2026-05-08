import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import time
import json
import math
from hgdm_ultimate import HGDMUltimate, HGDMConfig
from utils import get_gpu_memory_usage

def measure_state_stability():
    device = torch.device('cuda')
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    model = HGDMUltimate(config).to(device)
    model.eval()
    
    gen_len = 100000
    log_interval = 1000
    
    print(f"\n{'='*50}\nExp 10: State Stability (100k Token Stress Test)\n{'='*50}")
    
    norms = []
    current_state = None
    
    # Starting token
    x = torch.randint(0, 256, (1, 1), device=device)
    
    t_start = time.time()
    
    for i in range(gen_len):
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, current_state = model(x, states=current_state)
                # Sample next token
                x = torch.argmax(logits[:, -1:], dim=-1)
        
        if i % log_interval == 0:
            # Calculate Frobenius Norm of the states across all layers
            # current_state is a list of [B, H, DK, DV] tensors
            total_norm = 0
            for layer_state in current_state:
                total_norm += torch.norm(layer_state).item()
            
            avg_norm = total_norm / len(current_state)
            vram = get_gpu_memory_usage()
            print(f"Step {i:6d} | Avg State Norm: {avg_norm:10.2f} | VRAM: {vram:5.0f}MB")
            norms.append({"step": i, "avg_norm": avg_norm, "vram_mb": vram})
            
    elapsed = time.time() - t_start
    print(f"\n100k Generation Complete in {elapsed:.1f}s")
    
    with open("results.json", "w") as f:
        json.dump({"history": norms, "total_time_s": elapsed}, f, indent=4)
        
    print("Saved results.json")

if __name__ == "__main__":
    measure_state_stability()
