import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import time
import json
import math
from hgdm_ultimate import HGDMUltimate, HGDMConfig

def generate_synthetic_math_data(num_samples=10000, seq_len=512):
    """Generates basic arithmetic and algebra sequences for a quick domain transfer proof."""
    print("Generating synthetic math domain data...")
    text = ""
    for i in range(num_samples):
        a, b = torch.randint(1, 1000, (2,)).tolist()
        text += f"Problem: {a} + {b} = ?\nSolution: The sum of {a} and {b} is {a+b}.\n\n"
        
        c = torch.randint(1, 100, (1,)).item()
        d = torch.randint(1, 100, (1,)).item()
        text += f"Solve for x: {c}x = {c*d}\nx = {c*d} / {c}\nx = {d}\n\n"
        
    data = torch.tensor(list(text.encode('utf-8')), dtype=torch.uint8).long()
    print(f"Generated {len(data)/1024**2:.1f} MB of synthetic math text.")
    return data

def train_math_transfer(model, train_data, steps=500, seq_len=512):
    device = torch.device('cuda')
    model.train()
    
    lr = 2e-4 # Lower learning rate for fine-tuning
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    
    print("\n--- Fine-Tuning HGDM on Math Domain ---")
    history = []
    
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        
        ix = torch.randint(len(train_data) - seq_len - 1, (1,))
        x = torch.stack([train_data[i:i+seq_len] for i in ix]).to(device)
        y = torch.stack([train_data[i+1:i+seq_len+1] for i in ix]).to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
            if isinstance(out, tuple): out = out[0]
            loss = F.cross_entropy(out.view(-1, 256), y.view(-1))
            
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        
        if step % 50 == 0:
            bpb = loss.item() / math.log(2)
            print(f"Step {step:4d} | Math Domain BPB: {bpb:.4f}")
            history.append({"step": step, "bpb": bpb})
            
    return history

if __name__ == "__main__":
    device = torch.device('cuda')
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    model = HGDMUltimate(config).to(device)
    
    checkpoint_path = "hgdm_enwik8_120M.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading base Enwik8 checkpoint from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    else:
        print(f"WARNING: Base checkpoint not found. Training from scratch.")
        
    math_data = generate_synthetic_math_data()
    
    # Evaluate BEFORE fine-tuning (Zero-shot domain transfer)
    print("\nEvaluating Zero-Shot Math Performance...")
    model.eval()
    with torch.no_grad():
        ix = torch.randint(len(math_data) - 512 - 1, (10,))
        x = torch.stack([math_data[i:i+512] for i in ix]).to(device)
        y = torch.stack([math_data[i+1:i+512+1] for i in ix]).to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)[0]
            initial_loss = F.cross_entropy(out.view(-1, 256), y.view(-1)).item()
            initial_bpb = initial_loss / math.log(2)
    print(f"Initial Math BPB: {initial_bpb:.4f}")
    
    # Fine-tune
    history = train_math_transfer(model, math_data, steps=500, seq_len=512)
    
    results = {
        "initial_math_bpb": initial_bpb,
        "finetune_history": history
    }
    
    with open("results_exp6_math.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nExperiment 6 Complete. Saved results_exp6_math.json")
