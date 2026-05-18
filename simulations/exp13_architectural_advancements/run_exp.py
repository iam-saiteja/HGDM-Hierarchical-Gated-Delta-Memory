"""
Exp 13: Architectural Advancements Benchmark
=============================================
Compares the baseline HGDM against the Advanced HGDM model featuring:
  1. Variable-Delta-t Continuous-Time Gating (Feature 2)
  2. Cross-Layer State Fusion / State Highways (Feature 6)

Trains both models for 300 steps on Enwik8 under identical conditions.
"""

import torch
import torch.nn.functional as F
import time
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hgdm_ultimate import HGDMUltimate, HGDMConfig
from simulations.utils import get_enwik8_data

def train_model(config, name, train_data, steps=300, seq_len=512, batch_size=4):
    device = torch.device('cuda')
    model = HGDMUltimate(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n--- Training {name} ---")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")
    
    losses = []
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats()
    
    model.train()
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        
        # Draw random batch
        ix = torch.randint(len(train_data) - seq_len - 1, (batch_size,))
        x = torch.stack([train_data[i:i+seq_len] for i in ix]).to(device)
        y = torch.stack([train_data[i+1:i+seq_len+1] for i in ix]).to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
            
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        
        losses.append(loss.item())
        
        if step % 50 == 0:
            print(f"  step {step:4d} | loss {loss.item():.4f}")
            
    total_time = time.time() - t_start
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    throughput = (steps * batch_size * seq_len) / total_time
    
    print(f"Done. Time: {total_time:.1f}s | Speed: {throughput:.0f} tok/s | Peak VRAM: {peak_vram:.1f} MB")
    
    return {
        "final_loss": losses[-1],
        "mean_loss_last_50": sum(losses[-50:]) / 50,
        "time": total_time,
        "throughput": throughput,
        "peak_vram": peak_vram,
        "losses": losses
    }

def main():
    if not torch.cuda.is_available():
        print("CUDA required for running exp13."); return
        
    device = torch.device('cuda')
    train_data, _ = get_enwik8_data()
    
    # 1. Configs
    cfg_baseline = HGDMConfig(
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_k=64,
        d_v=64,
        d_ff=512,
        vocab_size=256,
        use_variable_delta_t=False,
        use_state_fusion=False
    )
    
    cfg_advanced = HGDMConfig(
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_k=64,
        d_v=64,
        d_ff=512,
        vocab_size=256,
        use_variable_delta_t=True,
        use_state_fusion=True
    )
    
    print("=" * 65)
    print("EXP 13: ADVANCED HGDM ARCHITECTURAL COMPARISON")
    print("=" * 65)
    
    # Run trainings
    baseline_stats = train_model(cfg_baseline, "Baseline HGDM", train_data)
    advanced_stats = train_model(cfg_advanced, "Advanced HGDM (CT-Decay + State Highways)", train_data)
    
    # 2. Render comparative analysis
    print("\n" + "=" * 65)
    print("COMPARATIVE SUMMARY TABLE")
    print("=" * 65)
    print(f"{'Metric':<25} | {'Baseline HGDM':<15} | {'Advanced HGDM':<15} | {'Change':<10}")
    print("-" * 65)
    
    def print_row(label, val_base, val_adv, fmt, higher_better=False):
        change = ((val_adv - val_base) / val_base) * 100
        sign = "+" if change >= 0 else ""
        color = "🟢" if (change >= 0 if higher_better else change <= 0) else "🔴"
        print(f"{label:<25} | {val_base:<15{fmt}} | {val_adv:<15{fmt}} | {color} {sign}{change:.1f}%")
        
    print_row("Final Step Loss", baseline_stats["final_loss"], advanced_stats["final_loss"], ".4f", higher_better=False)
    print_row("Mean Loss (Last 50 Steps)", baseline_stats["mean_loss_last_50"], advanced_stats["mean_loss_last_50"], ".4f", higher_better=False)
    print_row("Training Time (s)", baseline_stats["time"], advanced_stats["time"], ".2f", higher_better=False)
    print_row("Throughput (tokens/s)", baseline_stats["throughput"], advanced_stats["throughput"], ".0f", higher_better=True)
    print_row("Peak VRAM Allocated (MB)", baseline_stats["peak_vram"], advanced_stats["peak_vram"], ".1f", higher_better=False)
    print("=" * 65)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    results = {
        "baseline": baseline_stats,
        "advanced": advanced_stats
    }
    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved in results/results.json")

if __name__ == "__main__":
    main()
