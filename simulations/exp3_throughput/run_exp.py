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

def measure_throughput():
    device = torch.device('cuda')
    lengths = [512, 1024, 2048, 4096, 8192]
    
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    hgdm = HGDMUltimate(config).to(device)
    
    # We use FlashAttention=True for Transformer to show its best-case scenario
    transformer = BaselineTransformer(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256, use_flash=True).to(device)
    
    hg_speed = []
    tf_speed = []
    hg_vram = []
    tf_vram = []
    
    print(f"\n--- Measuring Throughput (Tokens/Sec) ---")
    
    for L in lengths:
        x = torch.randint(0, 256, (1, L), device=device)
        
        # HGDM Warmup & Bench
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5): hgdm(x)
        t0 = time.time()
        for _ in range(20): _ = hgdm(x)
        t_hg = (time.time() - t0) / 20
        speed_hg = L / t_hg
        vram_hg = torch.cuda.max_memory_allocated() / (1024**2)
        hg_speed.append(speed_hg)
        hg_vram.append(vram_hg)
        
        # Transformer Warmup & Bench
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(5): transformer(x)
            t0 = time.time()
            for _ in range(20): _ = transformer(x)
            t_tf = (time.time() - t0) / 20
            speed_tf = L / t_tf
            vram_tf = torch.cuda.max_memory_allocated() / (1024**2)
            tf_speed.append(speed_tf)
            tf_vram.append(vram_tf)
        except RuntimeError:
            speed_tf = 0
            vram_tf = 0
            tf_speed.append(0)
            tf_vram.append(0)
            
        print(f"L={L:5d} | HGDM: {speed_hg:8.0f} tok/s ({vram_hg:.0f}MB) | Trans: {speed_tf:8.0f} tok/s ({vram_tf:.0f}MB)")
        
    results = {
        "lengths": lengths,
        "HGDM_Tokens_Per_Sec": hg_speed,
        "HGDM_Peak_VRAM_MB": hg_vram,
        "Transformer_Tokens_Per_Sec": tf_speed,
        "Transformer_Peak_VRAM_MB": tf_vram
    }
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nExperiment 3 Complete. Saved results.json")

if __name__ == "__main__":
    measure_throughput()
