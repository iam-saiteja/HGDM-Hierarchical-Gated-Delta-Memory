import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import json
from utils import BaselineTransformer
from hgdm_ultimate import HGDMUltimate, HGDMConfig

def run_memory_test(model, name, seq_lens, flash_attention=False):
    device = torch.device('cuda')
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    results = {}
    test_name = f"{name} (FlashAttention={flash_attention})" if name == "Transformer" else name
    print(f"\n--- Testing Memory Scaling for {test_name} ---")
    
    for seq_len in seq_lens:
        print(f"Testing seq_len={seq_len}...", end=" ", flush=True)
        try:
            x = torch.randint(0, 256, (1, seq_len)).to(device)
            y = torch.randint(0, 256, (1, seq_len)).to(device)
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            for _ in range(20): # Run 20 steps to stabilize memory buffers
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    if name == "Transformer":
                        with torch.backends.cuda.sdp_kernel(enable_flash=flash_attention, enable_mem_efficient=flash_attention, enable_math=True):
                            out = model(x)
                    else:
                        out = model(x)
                        
                    if isinstance(out, tuple): out = out[0]
                    loss = F.cross_entropy(out.view(-1, 256), y.view(-1))
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            results[str(seq_len)] = peak_mem
            print(f"Survived. Peak VRAM: {peak_mem:.0f} MB")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("FAILED (OOM)")
                results[str(seq_len)] = "OOM"
                torch.cuda.empty_cache()
                if name == "Transformer":
                    # Transformers will OOM, no need to test larger sizes
                    break
            else:
                raise e
                
    return results

if __name__ == "__main__":
    seq_lens = [512, 1024, 2048, 4096, 8192, 16384]
    
    # Init Models
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    hgdm = HGDMUltimate(config)
    transformer = BaselineTransformer(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256, max_seq_len=16500)
    
    # 1. Transformer WITHOUT FlashAttention (Raw Math)
    tf_mem_no_flash = run_memory_test(transformer, "Transformer", seq_lens, flash_attention=False)
    
    # 2. Transformer WITH FlashAttention (Optimized Software)
    tf_mem_flash = run_memory_test(transformer, "Transformer", seq_lens, flash_attention=True)
    
    del transformer
    torch.cuda.empty_cache()
    
    # 3. HGDM (Constant Memory)
    hg_mem = run_memory_test(hgdm, "HGDM", seq_lens)
    
    results = {
        "Transformer_NoFlash_Memory_MB": tf_mem_no_flash,
        "Transformer_Flash_Memory_MB": tf_mem_flash,
        "HGDM_Memory_MB": hg_mem
    }
    
    with open("results_exp2.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nExperiment 2 Complete. Saved results_exp2.json")
