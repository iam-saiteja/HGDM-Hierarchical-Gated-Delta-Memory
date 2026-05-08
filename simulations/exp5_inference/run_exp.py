import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import json
import time
from hgdm_ultimate import HGDMUltimate, HGDMConfig
from utils import get_gpu_memory_usage

def run_long_inference():
    device = torch.device('cuda')
    
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    model = HGDMUltimate(config).to(device)
    
    checkpoint_path = "../exp1_enwik8/hgdm_enwik8_120M.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading trained checkpoint from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    else:
        print("WARNING: No checkpoint found. Generating with random weights.")
        
    model.eval()
    
    prompt_text = "The quick brown fox jumps over the lazy dog"
    prompt = torch.tensor([list(prompt_text.encode('utf-8'))], dtype=torch.long, device=device)
    
    gen_len = 2000
    print(f"\n--- Generating {gen_len} bytes from trained HGDM ---")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output_tensor = model.generate(prompt, max_new_bytes=gen_len, temp=0.8)[0]
            
    t1 = time.time()
    elapsed = t1 - t0
    speed = gen_len / elapsed
    sys_mem = get_gpu_memory_usage()
    
    text = bytes(output_tensor.cpu().tolist()).decode('utf-8', errors='ignore')
    
    print(f"\nGenerated Text Snippet:\n{text[:500]}...")
    print(f"\nInference Performance:")
    print(f"Time: {elapsed:.2f}s | Speed: {speed:.1f} bytes/s | VRAM: {sys_mem:.0f}MB")
    
    results = {
        "time_s": elapsed,
        "speed_bytes_s": speed,
        "vram_mb": sys_mem,
        "gen_len_bytes": gen_len,
        "text_sample": text
    }
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nExperiment 5 Complete. Saved results.json")

if __name__ == "__main__":
    run_long_inference()
