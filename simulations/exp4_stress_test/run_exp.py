import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import time
import json
from hgdm_ultimate import HGDMUltimate, HGDMConfig

def run_stress_test():
    device = torch.device('cuda')
    # Extreme sequence lengths: 16k, 32k, 65k, 131k
    seq_lengths = [16384, 32768, 65536, 131072]
    batch_size = 1
    
    # Initialize 120M Baseline HGDM
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    print("Initializing HGDM-120M...")
    model = HGDMUltimate(config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    results = {"seq_lengths": seq_lengths, "memory": [], "status": []}
    
    print("\n--- HARDWARE STRESS TEST (LONG CONTEXT) ---")
    
    for seq_len in seq_lengths:
        print(f"Testing Extreme Length: {seq_len:,} tokens...", end=" ", flush=True)
        try:
            x = torch.randint(0, 256, (batch_size, seq_len)).to(device)
            y = torch.randint(0, 256, (batch_size, seq_len)).to(device)
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Run 5 forward/backward passes
            for _ in range(5):
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = model(x)
                    loss = nn.CrossEntropyLoss()(logits.view(-1, 256), y.view(-1))
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            results["memory"].append(peak_mem)
            results["status"].append("SUCCESS")
            print(f"SUCCESS | Peak VRAM: {peak_mem:.0f}MB")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("FAILED (OOM)")
                results["memory"].append(None)
                results["status"].append("OOM")
                torch.cuda.empty_cache()
            else:
                print(f"FAILED (Error: {e})")
                results["memory"].append(None)
                results["status"].append(str(e))
                torch.cuda.empty_cache()
                
    with open("stress_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved stress_results.json")

if __name__ == "__main__":
    run_stress_test()
