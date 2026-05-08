import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import time
import json
import math
import urllib.request
import zipfile
from hgdm_ultimate import HGDMUltimate, HGDMConfig

def get_data():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "enwik8.zip")
    data_path = os.path.join(data_dir, "enwik8")
    
    if not os.path.exists(data_path):
        if not os.path.exists(zip_path):
            print("Downloading enwik8 (100MB)... This may take a minute.")
            url = "http://mattmahoney.net/dc/enwik8.zip"
            urllib.request.urlretrieve(url, zip_path)
        print("Extracting enwik8...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
    with open(data_path, 'rb') as f:
        data = f.read()
    
    n = len(data)
    train_data = torch.frombuffer(data[:int(n * 0.9)], dtype=torch.uint8).long()
    return train_data

def set_ablation_mode(model, mode):
    H = model.config.n_heads
    if mode == 'full': return
    
    if mode == 'flat':
        tau = 200.0
        alpha_target = math.exp(-1.0 / tau)
        bias_val = math.log(alpha_target / (1.0 - alpha_target + 1e-8))
        biases = [bias_val] * H
    elif mode == 'learned':
        biases = [0.0] * H

    for layer in model.layers:
        mixer = layer.mixer
        with torch.no_grad():
            mixer.W_alpha.bias.copy_(torch.tensor(biases))

def train_ablation(mode, train_data, max_steps=2000):
    device = torch.device('cuda')
    seq_len = 2048
    micro_batch = 1
    accum_steps = 4 # Gradient accumulation for stability
    
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    model = HGDMUltimate(config).to(device)
    set_ablation_mode(model, mode)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=4e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    model.train()
    
    losses = []
    bpbs = []
    
    print(f"\n--- Training {mode.upper()} Gating on Enwik8 ---")
    start_time = time.time()
    
    for step in range(max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        for _ in range(accum_steps):
            ix = torch.randint(len(train_data) - seq_len - 1, (micro_batch,))
            x = torch.stack([train_data[i:i+seq_len] for i in ix]).to(device)
            y = torch.stack([train_data[i+1:i+seq_len+1] for i in ix]).to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = nn.CrossEntropyLoss()(logits.view(-1, 256), y.view(-1)) / accum_steps
                
            scaler.scale(loss).backward()
            accum_loss += loss.item() * accum_steps
            
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        avg_loss = accum_loss / accum_steps
        bpb = avg_loss / math.log(2)
        
        if step % 100 == 0:
            print(f"Step {step} | Loss: {avg_loss:.4f} | BPB: {bpb:.4f}")
            losses.append((step, avg_loss))
            bpbs.append((step, bpb))
            
    print(f"Finished {mode.upper()} in {time.time()-start_time:.1f}s")
    return {"losses": losses, "bpbs": bpbs}

if __name__ == "__main__":
    train_data = get_data()
    results = {}
    for mode in ['full', 'flat', 'learned']:
        results[mode] = train_ablation(mode, train_data)
        
    with open("ablation_enwik8_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved ablation_enwik8_results.json")
