import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
from benchmarks.architecture.hgdm_v3 import HGDMConfig, HGDMPatch

def train(steps=5000, batch_size=16, seq_len=256, lr=4e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Training HGDM-v3 on {device} ---")
    
    if not os.path.exists("input.txt"):
        print("Downloading input.txt...")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, "input.txt")
    
    with open("input.txt", 'r', encoding='utf-8') as f:
        data = f.read().encode('utf-8')
    data = torch.tensor(list(data), dtype=torch.long, device=device)
    
    split = int(0.9 * len(data))
    train_data = data[:split]
    
    config = HGDMConfig()
    model = HGDMPatch(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    
    def get_batch(split_data):
        ix = torch.randint(len(split_data) - seq_len - 1, (batch_size,))
        x = torch.stack([split_data[i:i+seq_len] for i in ix])
        y = torch.stack([split_data[i+1:i+seq_len+1] for i in ix])
        return x.to(device), y.to(device)

    t0 = time.time()
    best_loss = float('inf')
    
    for step in range(steps + 1):
        x, y = get_batch(train_data)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        if step % 100 == 0:
            elapsed = time.time() - t0
            bpb = loss.item() / math.log(2)
            print(f"Step {step:4d} | BPB: {bpb:.4f} | Time: {elapsed:.1f}s")
            t0 = time.time()
            
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), "hgdm_v3.pt")

    print("\nTraining complete. Weights saved to hgdm_v3.pt")

if __name__ == "__main__":
    train()
