import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
import time
import json
from utils import BaselineTransformer
from hgdm_ultimate import HGDMUltimate, HGDMConfig

def run_throughput_test(model, name, seq_lens):
    device = torch.device('cuda')
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    results = {}
    print(f"\n--- Testing Throughput for {name} ---")
    
    for seq_len in seq_lens:
        print(f"Testing seq_len={seq_len}...", end=" ", flush=True)
        try:
            x = torch.randint(0, 256, (1, seq_len)).to(device)
            y = torch.randint(0, 256, (1, seq_len)).to(device)
            
            # Warmup
            for _ in range(10):
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = model(x)
                    if isinstance(out, tuple): out = out[0]
                    loss = F.cross_entropy(out.view(-1, 256), y.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            torch.cuda.synchronize()
            t0 = time.time()
            
            iters = 50
            for _ in range(iters):
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = model(x)
                    if isinstance(out, tuple): out = out[0]
                    loss = F.cross_entropy(out.view(-1, 256), y.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            torch.cuda.synchronize()
            t1 = time.time()
            
            total_time = t1 - t0
            tokens_per_sec = (iters * seq_len) / total_time
            results[str(seq_len)] = tokens_per_sec
            print(f"{tokens_per_sec:.0f} tokens/s")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("FAILED (OOM)")
                results[str(seq_len)] = "OOM"
                torch.cuda.empty_cache()
                if name == "Transformer":
                    break
            else:
                raise e
                
    return results

if __name__ == "__main__":
    seq_lens = [512, 1024, 2048, 4096, 8192]
    
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    hgdm = HGDMUltimate(config)
    transformer = BaselineTransformer(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256, max_seq_len=16500)
    
    tf_speed = run_throughput_test(transformer, "Transformer (with FlashAttention)", seq_lens)
    
    del transformer
    torch.cuda.empty_cache()
    
    hg_speed = run_throughput_test(hgdm, "HGDM (Fused Triton Kernel)", seq_lens)
    
    results = {
        "Transformer_Tokens_Per_Sec": tf_speed,
        "HGDM_Tokens_Per_Sec": hg_speed
    }
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nExperiment 3 Complete. Saved results.json")
