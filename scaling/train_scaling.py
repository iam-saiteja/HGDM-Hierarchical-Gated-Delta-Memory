import torch
import torch.nn.functional as F
import os
import sys
import json
import urllib.request
import zipfile
import math
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hgdm_omega import OmegaGDM, OmegaConfig

def get_enwik8_data():
    """Downloads and returns Enwik8 data splits as long tensors."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "enwik8.zip")
    data_path = os.path.join(data_dir, "enwik8")
    
    if not os.path.exists(data_path):
        if not os.path.exists(zip_path):
            print("[Data] Downloading enwik8 (100MB)...")
            url = "http://mattmahoney.net/dc/enwik8.zip"
            urllib.request.urlretrieve(url, zip_path)
        print("[Data] Extracting enwik8...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
    with open(data_path, 'rb') as f:
        data = f.read()
    
    n = len(data)
    train_data = torch.frombuffer(data[:int(n * 0.9)], dtype=torch.uint8).long().clone()
    val_data = torch.frombuffer(data[int(n * 0.9):], dtype=torch.uint8).long().clone()
    return train_data, val_data

@torch.no_grad()
def evaluate_model(model, val_data, seq_len=512, batches=100, batch_size=16, device='cuda'):
    """Computes the average cross-entropy loss over random validation slices."""
    model.eval()
    total_loss = 0.0
    for _ in range(batches):
        ix = torch.randint(len(val_data) - seq_len - 1, (batch_size,))
        x = torch.stack([val_data[i:i+seq_len] for i in ix]).to(device)
        y = torch.stack([val_data[i+1:i+seq_len+1] for i in ix]).to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            total_loss += loss.item()
            
    model.train()
    return total_loss / batches

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def train_scaling():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_data, val_data = get_enwik8_data()
    
    # 3000 steps per model to ensure proper convergence for larger models
    steps = 3000
    batch_size = 16
    seq_len = 512
    warmup_steps = int(steps * 0.1)
    
    configs = [
        {"name": "10M", "lr": 1e-3, "cfg": OmegaConfig(d_byte=128, catcher_layers=1, renderer_layers=1, d_model=128, core_layers=4, n_heads=4, d_k=32, d_v=32, d_ff=512, decimation_rate=4, max_position_embeddings=512, vocab_size=256, use_state_fusion=False)},
        {"name": "35M", "lr": 6e-4, "cfg": OmegaConfig(d_byte=256, catcher_layers=2, renderer_layers=2, d_model=256, core_layers=6, n_heads=8, d_k=32, d_v=32, d_ff=1024, decimation_rate=8, max_position_embeddings=512, vocab_size=256, use_state_fusion=False)},
        {"name": "120M", "lr": 3e-4, "cfg": OmegaConfig(d_byte=256, catcher_layers=2, renderer_layers=2, d_model=768, core_layers=12, n_heads=12, d_k=64, d_v=64, d_ff=3072, decimation_rate=8, max_position_embeddings=512, vocab_size=256, use_state_fusion=False)}
    ]
    
    results = {}
    
    for c in configs:
        name = c["name"]
        lr = c["lr"]
        print(f"\n=========================================")
        print(f"Starting Training for Scale: {name}")
        print(f"Max LR: {lr} | Steps: {steps}")
        print(f"=========================================")
        
        model = OmegaGDM(c["cfg"], force_sequential=False).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = get_scheduler(optimizer, warmup_steps, steps)
        
        param_count = count_parameters(model)
        print(f"Parameters: {param_count:,}")
        
        model.train()
        pbar = tqdm(range(steps), desc=f"Training {name}")
        for step in pbar:
            ix = torch.randint(len(train_data) - seq_len - 1, (batch_size,))
            x = torch.stack([train_data[i:i+seq_len] for i in ix]).to(device)
            y = torch.stack([train_data[i+1:i+seq_len+1] for i in ix]).to(device)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            if step % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.2e}"})
                
        # Final evaluation
        print(f"\nRunning final validation for {name}...")
        val_loss = evaluate_model(model, val_data, seq_len=seq_len, batches=100, batch_size=batch_size, device=device)
        bpb = val_loss / math.log(2)
        print(f"Final Val Loss: {val_loss:.4f} | BPB: {bpb:.4f}")
        
        results[name] = {
            "parameters": param_count,
            "val_loss": val_loss,
            "bpb": bpb
        }
        
        # Save checkpoint
        ckpt_path = os.path.join(os.path.dirname(__file__), f"omega_scale_{name.lower()}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")
        
        # Free memory
        del model
        del optimizer
        del scheduler
        torch.cuda.empty_cache()
        
    out_path = os.path.join(os.path.dirname(__file__), "scaling_data.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nScaling Benchmark Complete! Data saved to {out_path}")

if __name__ == "__main__":
    train_scaling()
