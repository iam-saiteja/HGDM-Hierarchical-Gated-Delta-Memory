import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import time
import json
import math
from hgdm_ultimate import HGDMUltimate, HGDMConfig
from utils import get_enwik8_data, evaluate_model, get_gpu_memory_usage

def train_long_gating(mode, train_data, steps=1000, seq_len=4096):
    device = torch.device('cuda')
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    model = HGDMUltimate(config).to(device)
    
    # Apply gating override if flat
    if mode == 'flat':
        for layer in model.layers:
            tau = 200.0
            alpha_target = math.exp(-1.0 / tau)
            bias_val = math.log(alpha_target / (1.0 - alpha_target + 1e-8))
            layer.mixer.W_alpha.bias.data.fill_(bias_val)
            layer.mixer.W_alpha.bias.requires_grad = False
            
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n--- Training Long Gating ({mode.upper()}) | Seq: {seq_len} ---")
    
    history = []
    t_start = time.time()
    
    for step in range(steps + 1):
        idx = torch.randint(0, train_data.size(0) - seq_len - 1, (1,))
        x = train_data[idx:idx+seq_len].unsqueeze(0).to(device)
        y = train_data[idx+1:idx+seq_len+1].unsqueeze(0).to(device)
        
        opt.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = nn.CrossEntropyLoss()(logits.view(-1, 256), y.view(-1))
            
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
        if step % 100 == 0:
            bpb = loss.item() / math.log(2)
            vram = get_gpu_memory_usage()
            elapsed = time.time() - t_start
            print(f"Step {step:4d} | BPB: {bpb:.4f} | VRAM: {vram:5.0f}MB | Time: {elapsed:.1f}s")
            history.append({"step": step, "bpb": bpb, "vram": vram})
            
    return history

def run_experiment():
    train_data, val_data = get_enwik8_data()
    results = {}
    
    # Run Full (Hierarchical)
    results["full"] = train_long_gating("full", train_data)
    
    # Run Flat
    results["flat"] = train_long_gating("flat", train_data)
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nExperiment 9 Complete.")

if __name__ == "__main__":
    run_experiment()
