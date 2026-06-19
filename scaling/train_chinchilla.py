import torch
import torch.nn.functional as F
import os
import sys
import json
import math
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hgdm_omega import OmegaGDM, OmegaConfig

class ByteStreamDataset(IterableDataset):
    def __init__(self, hf_dataset, seq_len=512):
        self.dataset = hf_dataset
        self.seq_len = seq_len

    def __iter__(self):
        buffer = []
        for item in self.dataset:
            text = item['text']
            buffer.extend(list(text.encode('utf-8', errors='ignore')))
            while len(buffer) > self.seq_len + 1:
                x = buffer[:self.seq_len]
                y = buffer[1:self.seq_len+1]
                buffer = buffer[self.seq_len:]
                yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

@torch.no_grad()
def evaluate_model(model, val_loader, batches=50, device='cuda'):
    model.eval()
    total_loss = 0.0
    
    val_iter = iter(val_loader)
    for _ in range(batches):
        try:
            x, y = next(val_iter)
        except StopIteration:
            break
        
        x, y = x.to(device), y.to(device)
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
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def train_chinchilla():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # HuggingFace streaming dataset to avoid 38GB download
    print("Connecting to HuggingFace OpenWebText (Streaming Mode)...")
    ds_train = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    # We will just skip the first 10,000 documents for validation to ensure no overlap
    ds_val = ds_train.take(10000)
    ds_train = ds_train.skip(10000)
    
    seq_len = 512
    batch_size = 64
    grad_accum_steps = 4
    tokens_per_update = batch_size * grad_accum_steps * seq_len
    
    train_loader = DataLoader(ByteStreamDataset(ds_train, seq_len), batch_size=batch_size, num_workers=0)
    val_loader = DataLoader(ByteStreamDataset(ds_val, seq_len), batch_size=batch_size, num_workers=0)
    
    configs = [
        {"name": "10M", "lr": 1e-3, "cfg": OmegaConfig(d_byte=128, catcher_layers=1, renderer_layers=1, d_model=128, core_layers=4, n_heads=4, d_k=32, d_v=32, d_ff=512, decimation_rate=4, max_position_embeddings=512, vocab_size=256, use_state_fusion=False)},
        {"name": "35M", "lr": 6e-4, "cfg": OmegaConfig(d_byte=256, catcher_layers=2, renderer_layers=2, d_model=256, core_layers=6, n_heads=8, d_k=32, d_v=32, d_ff=1024, decimation_rate=8, max_position_embeddings=512, vocab_size=256, use_state_fusion=False)},
        {"name": "120M", "lr": 3e-4, "cfg": OmegaConfig(d_byte=256, catcher_layers=2, renderer_layers=2, d_model=768, core_layers=12, n_heads=12, d_k=64, d_v=64, d_ff=3072, decimation_rate=8, max_position_embeddings=512, vocab_size=256, use_state_fusion=False)}
    ]
    
    results = {}
    
    for c in configs:
        model = OmegaGDM(c["cfg"], force_sequential=False).to(device)
        param_count = count_parameters(model)
        
        # CHINCHILLA COMPUTE-OPTIMAL CALCULATION: 20x Parameter Count
        target_tokens = param_count * 20
        total_steps = math.ceil(target_tokens / tokens_per_update)
        warmup_steps = int(total_steps * 0.1)
        
        name = c["name"]
        lr = c["lr"]
        print(f"\n" + "="*50)
        print(f"Starting True Chinchilla Run: {name}")
        print(f"Parameters: {param_count:,}")
        print(f"Target Tokens (20x): {target_tokens:,}")
        print(f"Total Updates Required: {total_steps:,} (Tokens/Update: {tokens_per_update:,})")
        print("="*50)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = get_scheduler(optimizer, warmup_steps, total_steps)
        
        model.train()
        train_iter = iter(train_loader)
        
        pbar = tqdm(range(total_steps), desc=f"Training {name}")
        for step in pbar:
            for _ in range(grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)
                    
                x, y = x.to(device), y.to(device)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
                    loss = loss / grad_accum_steps
                
                loss.backward()
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            if step % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({'loss': f"{loss.item() * grad_accum_steps:.4f}", 'lr': f"{current_lr:.2e}"})
                
            # Checkpoint the large model to be safe
            if name == "120M" and step > 0 and step % 2000 == 0:
                torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), f"omega_{name}_step{step}.pt"))
                
        # Final evaluation
        print(f"\nRunning final validation for {name}...")
        val_loss = evaluate_model(model, val_loader, batches=100, device=device)
        bpb = val_loss / math.log(2)
        print(f"Final Val Loss: {val_loss:.4f} | BPB: {bpb:.4f}")
        
        results[name] = {
            "parameters": param_count,
            "tokens_trained": target_tokens,
            "val_loss": val_loss,
            "bpb": bpb
        }
        
        ckpt_path = os.path.join(os.path.dirname(__file__), f"omega_chinchilla_{name.lower()}.pt")
        torch.save(model.state_dict(), ckpt_path)
        
        del model
        del optimizer
        del scheduler
        torch.cuda.empty_cache()
        
    out_path = os.path.join(os.path.dirname(__file__), "chinchilla_scaling_data.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nTrue Chinchilla Benchmark Complete! Data saved to {out_path}")

if __name__ == "__main__":
    train_chinchilla()
