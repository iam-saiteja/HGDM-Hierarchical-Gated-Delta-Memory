import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
import time
import json
import math
from utils import get_enwik8_data, get_gpu_memory_usage
from hgdm_ultimate import HGDMUltimate, HGDMConfig

def train_ablation(mode, train_data, steps=2000, seq_len=1024):
    device = torch.device('cuda')
    
    # Configure Gating Modes
    # full: hierarchical multi-scale
    # flat: static tau=200
    # learned: no tau bias
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    model = HGDMUltimate(config).to(device)
    
    # Apply ablation overrides
    for layer in model.layers:
        if mode == 'flat':
            # Force all tau to be exactly 200 (medium scale)
            layer.tau_init.data.fill_(200.0)
            layer.tau_init.requires_grad = False
        elif mode == 'learned':
            # Let the model learn everything from scratch without the hierarchical bias
            layer.tau_init.data.fill_(100.0)
            layer.tau_init.requires_grad = True
            
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n--- Training Ablation Mode: {mode.upper()} ---")
    history = []
    t_start = time.time()
    
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        
        ix = torch.randint(len(train_data) - seq_len - 1, (1,))
        x = torch.stack([train_data[i:i+seq_len] for i in ix]).to(device)
        y = torch.stack([train_data[i+1:i+seq_len+1] for i in ix]).to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
            if isinstance(out, tuple): out = out[0]
            loss = F.cross_entropy(out.view(-1, 256), y.view(-1))
            
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
        if step % 100 == 0:
            bpb = loss.item() / math.log(2)
            sys_mem = get_gpu_memory_usage()
            elapsed = time.time() - t_start
            print(f"Step {step:4d} | BPB: {bpb:.4f} | VRAM: {sys_mem:.0f}MB | Time: {elapsed:.1f}s")
            history.append({
                "step": step,
                "bpb": bpb,
                "vram_mb": sys_mem,
                "time_s": elapsed
            })
            
    # Generation Proof
    print(f"--- Generating sample from {mode} ---")
    model.eval()
    prompt = torch.tensor([list("In the beginning ".encode('utf-8'))], dtype=torch.long, device=device)
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output_tensor = model.generate(prompt, max_new_bytes=512, temp=0.8)[0]
    
    return {
        "history": history,
        "sample": bytes(output_tensor.cpu().tolist()).decode('utf-8', errors='ignore')
    }

if __name__ == "__main__":
    train_data, _ = get_enwik8_data()
    
    results = {}
    for mode in ['full', 'flat', 'learned']:
        results[mode] = train_ablation(mode, train_data)
        
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nExperiment 4 Complete. Saved results.json")
