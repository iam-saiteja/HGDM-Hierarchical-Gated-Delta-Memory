import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
import time
import json
from hgdm_ultimate import HGDMUltimate, HGDMConfig
from utils import BaselineTransformer, get_gpu_memory_usage

def measure_memory_scaling():
    device = torch.device('cuda')
    lengths = [512, 1024, 2048, 4096, 8192, 16384]
    
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    hgdm = HGDMUltimate(config).to(device)
    
    # Transformer with FlashAttention disabled to show O(N^2) explosion
    transformer = BaselineTransformer(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256, use_flash=False).to(device)
    
    hg_vram = []
    tf_vram = []
    
    print(f"\n--- Measuring Memory Scaling ---")
    
    for L in lengths:
        # HGDM Measurement
        torch.cuda.empty_cache()
        x = torch.randint(0, 256, (1, L), device=device)
        
        try:
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _ = hgdm(x)
            sys_mem = get_gpu_memory_usage()
            hg_vram.append(sys_mem)
            print(f"HGDM | L={L:5d} | VRAM: {sys_mem:.0f}MB")
        except RuntimeError as e:
            print(f"HGDM | L={L:5d} | OOM")
            hg_vram.append(None)
            
        # Transformer Measurement
        torch.cuda.empty_cache()
        
        try:
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _ = transformer(x)
            sys_mem = get_gpu_memory_usage()
            tf_vram.append(sys_mem)
            print(f"Trans | L={L:5d} | VRAM: {sys_mem:.0f}MB")
        except RuntimeError as e:
            print(f"Trans | L={L:5d} | OOM")
            tf_vram.append(None)
            
    results = {
        "lengths": lengths,
        "HGDM_VRAM_MB": hg_vram,
        "Transformer_VRAM_MB": tf_vram
    }
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nExperiment 2 Complete. Saved results.json")

if __name__ == "__main__":
    measure_memory_scaling()
