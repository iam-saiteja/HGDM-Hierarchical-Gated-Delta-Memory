import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
import time
import json
import math
from hgdm_ultimate import HGDMUltimate, HGDMConfig

def generate_math_problem():
    """Generates a synthetic algebraic equation string."""
    a = torch.randint(1, 100, (1,)).item()
    b = torch.randint(1, 100, (1,)).item()
    x = torch.randint(1, 10, (1,)).item()
    c = a * x + b
    return f"Solve for x: {a}x + {b} = {c}. Answer: x = {x}\n"

def train_math_transfer(steps=500, seq_len=128):
    device = torch.device('cuda')
    
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    model = HGDMUltimate(config).to(device)
    
    checkpoint_path = "../exp1_enwik8/hgdm_enwik8_120M.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading base Enwik8 checkpoint from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n--- Fine-tuning HGDM on Synthetic Math Domain ---")
    history = []
    t_start = time.time()
    
    for step in range(steps + 1):
        # Generate a batch of math problems
        problems = "".join([generate_math_problem() for _ in range(4)])
        data = torch.tensor(list(problems.encode('utf-8')), dtype=torch.long, device=device)
        
        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        
        # Take a slice
        x = data[:seq_len].unsqueeze(0)
        y = data[1:seq_len+1].unsqueeze(0)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
            if isinstance(out, tuple): out = out[0]
            loss = F.cross_entropy(out.view(-1, 256), y.view(-1))
            
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
        if step % 50 == 0:
            bpb = loss.item() / math.log(2)
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            current_mem = torch.cuda.memory_allocated() / (1024**2)
            elapsed = time.time() - t_start
            print(f"Step {step:4d} | Math BPB: {bpb:.4f} | Cur VRAM: {current_mem:.0f}MB | Peak: {peak_mem:.0f}MB | Time: {elapsed:.1f}s")
            history.append({
                "step": step,
                "bpb": bpb,
                "current_mem_mb": current_mem,
                "peak_mem_mb": peak_mem,
                "time_s": elapsed
            })

    # Inference Proof
    print(f"--- Generating Math Solution ---")
    model.eval()
    prompt = torch.tensor([list("Solve for x: 10x + 5 = 105. Answer: ".encode('utf-8'))], dtype=torch.long, device=device)
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output_tensor = model.generate(prompt, max_new_bytes=32, temp=0.5)[0]
    
    gen_text = bytes(output_tensor.cpu().tolist()).decode('utf-8', errors='ignore')
    print(f"Generated Result: {gen_text}")
    
    results = {
        "training": history,
        "inference": {
            "prompt": "Solve for x: 10x + 5 = 105. Answer: ",
            "generation": gen_text,
            "peak_mem_mb": torch.cuda.max_memory_allocated() / (1024**2)
        }
    }
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nExperiment 6 Complete. Saved results.json")

if __name__ == "__main__":
    train_math_transfer()
