import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
import time
import json
from hgdm_ultimate import HGDMUltimate, HGDMConfig
from utils import BaselineTransformer

def measure_memory_scaling():
    device = torch.device('cuda')
    lengths = [512, 1024, 2048, 4096, 8192, 16384]
    
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    hgdm = HGDMUltimate(config).to(device)
    
    # Transformer with FlashAttention disabled to show O(N^2) explosion
    transformer = BaselineTransformer(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256, use_flash=False).to(device)
    
    hg_mem_peak = []
    hg_mem_curr = []
    tf_mem_peak = []
    tf_mem_curr = []
    
    print(f"\n--- Measuring Memory Scaling ---")
    
    for L in lengths:
        # HGDM Measurement
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, 256, (1, L), device=device)
        
        try:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = hgdm(x)
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            curr = torch.cuda.memory_allocated() / (1024**2)
            hg_mem_peak.append(peak)
            hg_mem_curr.append(curr)
            print(f"HGDM | L={L:5d} | Cur VRAM: {curr:.0f}MB | Peak: {peak:.0f}MB")
        except RuntimeError as e:
            print(f"HGDM | L={L:5d} | OOM")
            hg_mem_peak.append(None)
            hg_mem_curr.append(None)
            
        # Transformer Measurement
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = transformer(x)
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            curr = torch.cuda.memory_allocated() / (1024**2)
            tf_mem_peak.append(peak)
            tf_mem_curr.append(curr)
            print(f"Trans | L={L:5d} | Cur VRAM: {curr:.0f}MB | Peak: {peak:.0f}MB")
        except RuntimeError as e:
            print(f"Trans | L={L:5d} | OOM")
            tf_mem_peak.append(None)
            tf_mem_curr.append(None)
            
    results = {
        "lengths": lengths,
        "HGDM_Peak_MB": hg_mem_peak,
        "HGDM_Current_MB": hg_mem_curr,
        "Transformer_Peak_MB": tf_mem_peak,
        "Transformer_Current_MB": tf_mem_curr
    }
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nExperiment 2 Complete. Saved results.json")

if __name__ == "__main__":
    measure_memory_scaling()
