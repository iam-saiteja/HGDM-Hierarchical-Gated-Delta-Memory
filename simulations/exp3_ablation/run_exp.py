import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import time
import json
import math
import urllib.request
from hgdm_ultimate import HGDMUltimate, HGDMConfig
from torch.utils.data import DataLoader, IterableDataset

# TinyShakespeare Dataloader (Fast for Ablations)
class ByteChunkDataset(IterableDataset):
    def __init__(self, data_bytes, seq_len):
        self.data = data_bytes
        self.seq_len = seq_len

    def __iter__(self):
        max_idx = len(self.data) - self.seq_len - 1
        while True:
            idx = torch.randint(0, max_idx, (1,)).item()
            chunk = self.data[idx:idx + self.seq_len + 1]
            x = torch.tensor(list(chunk[:-1]), dtype=torch.long)
            y = torch.tensor(list(chunk[1:]), dtype=torch.long)
            yield x, y

def get_dataloader(seq_len=1024, batch_size=4):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = "tinyshakespeare.txt"
    if not os.path.exists(filepath):
        print("Downloading TinyShakespeare...")
        urllib.request.urlretrieve(url, filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    data_bytes = text.encode('utf-8')
    dataset = ByteChunkDataset(data_bytes, seq_len)
    return DataLoader(dataset, batch_size=batch_size)

def set_ablation_mode(model, mode):
    """Overrides the default initialization to test gating ablations."""
    H = model.config.n_heads
    if mode == 'full': return # Baseline multi-scale
    
    if mode == 'flat':
        tau = 200.0
        alpha_target = math.exp(-1.0 / tau)
        bias_val = math.log(alpha_target / (1.0 - alpha_target + 1e-8))
        biases = [bias_val] * H
    elif mode == 'learned':
        # Zero bias means alpha starts at 0.5 for all heads, completely random/learned
        biases = [0.0] * H

    for layer in model.layers:
        mixer = layer.mixer
        with torch.no_grad():
            mixer.W_alpha.bias.copy_(torch.tensor(biases))

def train_ablation(mode, max_steps=2000):
    device = torch.device('cuda')
    seq_len = 1024
    batch_size = 4
    
    # 120M Params Configuration
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    model = HGDMUltimate(config).to(device)
    set_ablation_mode(model, mode)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    dataloader = get_dataloader(seq_len, batch_size)
    data_iter = iter(dataloader)
    
    model.train()
    scaler = torch.amp.GradScaler('cuda')
    
    losses = []
    bpbs = []
    
    print(f"\n--- Training {mode.upper()} Gating ---")
    start_time = time.time()
    for step in range(max_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
            
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        # BF16 for industrial stability
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = nn.CrossEntropyLoss()(logits.view(-1, 256), y.view(-1))
            
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        bpb = loss.item() / math.log(2)
        
        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | BPB: {bpb:.4f}")
            losses.append((step, loss.item()))
            bpbs.append((step, bpb))
            
    print(f"Finished {mode.upper()} in {time.time()-start_time:.1f}s")
    return {"losses": losses, "bpbs": bpbs}

if __name__ == "__main__":
    results = {}
    for mode in ['full', 'flat', 'learned']:
        results[mode] = train_ablation(mode)
        
    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved ablation_results.json")
