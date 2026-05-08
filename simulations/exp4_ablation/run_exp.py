import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
import time
import json
import math
from utils import get_enwik8_data
from hgdm_ultimate import HGDMUltimate, HGDMConfig

def set_ablation_mode(model, mode):
    H = model.config.n_heads
    if mode == 'full': return
    
    if mode == 'flat':
        tau = 200.0
        alpha_target = math.exp(-1.0 / tau)
        bias_val = math.log(alpha_target / (1.0 - alpha_target + 1e-8))
        biases = [bias_val] * H
    elif mode == 'learned':
        # Small random noise around sigmoid(0.5) = 0 bias
        biases = torch.randn(H).tolist()

    for layer in model.layers:
        mixer = layer.mixer
        with torch.no_grad():
            mixer.W_alpha.bias.copy_(torch.tensor(biases))

def train_ablation(mode, train_data, steps=2000):
    device = torch.device('cuda')
    seq_len = 2048
    micro_batch = 1
    accum_steps = 12
    
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    model = HGDMUltimate(config).to(device)
    set_ablation_mode(model, mode)
    
    lr = 4e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr/10)
    scaler = torch.amp.GradScaler('cuda')
    
    model.train()
    history = []
    
    print(f"\n--- Training {mode.upper()} Gating on Enwik8 ---")
    start_time = time.time()
    
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        for _ in range(accum_steps):
            ix = torch.randint(len(train_data) - seq_len - 1, (micro_batch,))
            x = torch.stack([train_data[i:i+seq_len] for i in ix]).to(device)
            y = torch.stack([train_data[i+1:i+seq_len+1] for i in ix]).to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, 256), y.view(-1)) / accum_steps
                
            scaler.scale(loss).backward()
            accum_loss += loss.item() * accum_steps
            
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        
        if step % 200 == 0:
            avg_loss = accum_loss / accum_steps
            bpb = avg_loss / math.log(2)
            print(f"Step {step:4d} | BPB: {bpb:.4f}")
            history.append({"step": step, "bpb": bpb})
            
    print(f"Finished {mode.upper()} in {time.time()-start_time:.1f}s")
    
    # Clean up to prevent VRAM fragmentation
    del model
    del opt
    del scaler
    torch.cuda.empty_cache()
    
    return history

if __name__ == "__main__":
    train_data, _ = get_enwik8_data()
    results = {}
    
    for mode in ['full', 'flat', 'learned']:
        results[mode] = train_ablation(mode, train_data)
        
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nExperiment 4 Complete. Saved results.json")
