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
                # For Exp 1, we let the Transformer use FlashAttention to give it a fair speed/memory chance at 2048 length.
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
            elapsed = time.time() - t_start
            
            val_bpb_str = "N/A"
            # Periodic validation evaluation
            if step % 200 == 0 or step == steps:
                val_loss = evaluate_model(model, val_data)
                val_bpb = val_loss / math.log(2)
                val_bpb_str = f"{val_bpb:.4f}"
            
            print(f"Step {step:4d} | Train BPB: {bpb:.4f} | Val BPB: {val_bpb_str} | Cur VRAM: {current_mem:.0f}MB | Peak: {peak_mem:.0f}MB | Time: {elapsed:.1f}s")
            history.append({
                "step": step,
                "train_bpb": bpb,
                "val_bpb": val_bpb if val_bpb_str != "N/A" else None,
                "current_mem_mb": current_mem,
                "peak_mem_mb": peak_mem,
                "time_s": elapsed
            })
            
    total_time = time.time() - t_start
    
    # Save checkpoint
    checkpoint_name = "transformer_enwik8_120M.pt" if "Transformer" in name else "hgdm_enwik8_120M.pt"
    torch.save(model.state_dict(), checkpoint_name)
    print(f"Saved checkpoint to {checkpoint_name}")
    
    return history, total_time

if __name__ == "__main__":
    train_data, val_data = get_enwik8_data()
    
    # Init Models
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    hgdm = HGDMUltimate(config)
    transformer = BaselineTransformer(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    
    # Train
    tf_history, tf_time = train_model(transformer, "Transformer Baseline", train_data, val_data)
    
    # Clear memory explicitly before HGDM
    del transformer
    torch.cuda.empty_cache()
    
    hg_history, hg_time = train_model(hgdm, "HGDM (Ours)", train_data, val_data)
    
    results = {
        "Transformer": {"history": tf_history, "total_time": tf_time},
        "HGDM": {"history": hg_history, "total_time": hg_time}
    }
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nExperiment 1 Complete. Saved results.json")
