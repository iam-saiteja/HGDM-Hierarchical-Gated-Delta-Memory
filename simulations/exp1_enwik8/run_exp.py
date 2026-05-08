import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
import time
import json
import math
from utils import get_enwik8_data, BaselineTransformer, evaluate_model
from hgdm_ultimate import HGDMUltimate, HGDMConfig

def train_model(model, name, train_data, val_data, steps=1000, micro_batch=1, accum_steps=12, seq_len=2048, lr=4e-4):
    device = torch.device('cuda')
    model.to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr/10)
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n{'='*50}\nStarting Exp 1: {name}\n{'='*50}")
    
    history = []
    t_start = time.time()
    
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        torch.cuda.reset_peak_memory_stats()
        
        for _ in range(accum_steps):
            ix = torch.randint(len(train_data) - seq_len - 1, (micro_batch,))
            x = torch.stack([train_data[i:i+seq_len] for i in ix]).to(device)
            y = torch.stack([train_data[i+1:i+seq_len+1] for i in ix]).to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(x)
                if isinstance(out, tuple): out = out[0]
                loss = F.cross_entropy(out.view(-1, 256), y.view(-1)) / accum_steps
                
            scaler.scale(loss).backward()
            accum_loss += loss.item() * accum_steps
            
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        
        if step % 50 == 0:
            avg_loss = accum_loss / accum_steps
            bpb = avg_loss / math.log(2)
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            current_mem = torch.cuda.memory_allocated() / (1024**2)
            reserved_mem = torch.cuda.memory_reserved() / (1024**2)
            elapsed = time.time() - t_start
            
            val_bpb_str = "N/A"
            if step % 50 == 0 or step == steps:
                val_loss = evaluate_model(model, val_data)
                val_bpb = val_loss / math.log(2)
                val_bpb_str = f"{val_bpb:.4f}"
            
            print(f"Step {step:4d} | Train BPB: {bpb:.4f} | Val BPB: {val_bpb_str} | Cur: {current_mem:.0f}MB | Res: {reserved_mem:.0f}MB | Peak: {peak_mem:.0f}MB | Time: {elapsed:.1f}s")
            history.append({
                "step": step,
                "train_bpb": bpb,
                "val_bpb": val_bpb if val_bpb_str != "N/A" else None,
                "current_mem_mb": current_mem,
                "reserved_mem_mb": reserved_mem,
                "peak_mem_mb": peak_mem,
                "time_s": elapsed
            })
            
    total_time = time.time() - t_start
    
    # Save checkpoint
    checkpoint_name = "transformer_enwik8_120M.pt" if "Transformer" in name else "hgdm_enwik8_120M.pt"
    torch.save(model.state_dict(), checkpoint_name)
    
    # =========================================================================
    # GENERATIVE INFERENCE PROOF
    # =========================================================================
    print(f"--- Generating sample from {name} ---")
    model.eval()
    prompt = torch.tensor([list("The ".encode('utf-8'))], dtype=torch.long, device=device)
    
    gen_len = 40000 if "HGDM" in name else 2000 # Generate 40KB for HGDM, 2KB for Transformer (it might OOM or slow down)
    torch.cuda.reset_peak_memory_stats()
    t_gen_start = time.time()
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            if "Transformer" in name:
                # Custom generation for standard Transformer (it doesn't have .generate in utils.py)
                output_tokens = prompt[0].tolist()
                for _ in range(gen_len):
                    inp = torch.tensor([output_tokens[-2048:]], device=device)
                    logits = model(inp)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                    output_tokens.append(next_token)
                output_tensor = torch.tensor(output_tokens)
            else:
                output_tensor = model.generate(prompt, max_new_bytes=gen_len, temp=0.8)[0]
                
    t_gen_end = time.time()
    gen_time = t_gen_end - t_gen_start
    gen_speed = gen_len / gen_time
    gen_peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    
    print(f"Inference: {gen_time:.1f}s | {gen_speed:.1f} bytes/s | {gen_peak_mem:.0f}MB peak\n")
    
    return {
        "training": history,
        "inference": {
            "time_s": gen_time,
            "speed_bytes_s": gen_speed,
            "peak_mem_mb": gen_peak_mem,
            "checkpoint": checkpoint_name
        }
    }

if __name__ == "__main__":
    train_data, val_data = get_enwik8_data()
    
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    hgdm = HGDMUltimate(config)
    transformer = BaselineTransformer(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    
    results = {}
    results["Transformer"] = train_model(transformer, "Transformer Baseline", train_data, val_data)
    
    del transformer
    torch.cuda.empty_cache()
    
    results["HGDM"] = train_model(hgdm, "HGDM (Ours)", train_data, val_data)
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nExperiment 1 Complete. Saved results.json")
