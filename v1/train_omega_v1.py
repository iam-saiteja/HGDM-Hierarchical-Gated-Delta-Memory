import torch
import torch.nn.functional as F
import time
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hgdm_omega import OmegaGDM, OmegaConfig
from v1.data_omega_v1 import get_omega_v1_dataloader

def train_omega_v1():
    parser = argparse.ArgumentParser(description="Train Omega Model Version 1")
    parser.add_argument("--steps", type=int, default=150000, help="Total training steps for 1 epoch of 4.5M samples")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=type(3e-4), default=3e-4, help="Peak learning rate")
    parser.add_argument("--block-size", type=int, default=2048, help="Sequence block length")
    parser.add_argument("--ckpt", type=str, default="omega_v1_latest.pt", help="Checkpoint save path")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Device selected: {device}")

    # Omega Model 1 Specification (~120M Parameters)
    omega_cfg = OmegaConfig(
        d_byte=256,
        catcher_layers=2,
        renderer_layers=2,
        d_model=768,
        core_layers=12,
        n_heads=12,
        d_k=64,
        d_v=64,
        d_ff=3072,
        decimation_rate=8,
        max_position_embeddings=args.block_size,
        vocab_size=256,
        use_state_fusion=False
    )
    
    print("[Model] Initializing Omega Model v1...")
    model = OmegaGDM(omega_cfg, force_sequential=False).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameter Count: {params/1e6:.2f}M")

    # Data Loader
    print("[Dataset] Initializing OpenHermes-2.5 stream with Identity Injection...")
    dataloader = get_omega_v1_dataloader(
        batch_size=args.batch_size, 
        block_size=args.block_size
    )
    data_stream = iter(dataloader)

    # Optimizer & Scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    start_step = 0
    if os.path.exists(args.ckpt):
        print(f"[System] Resuming from checkpoint {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        start_step = ckpt['step']
        scaler.load_state_dict(ckpt['scaler'])

    print("=========================================")
    print(f"STARTING OMEGA V1 RUN (Target: {args.steps} steps)")
    print("=========================================")
    model.train()
    
    t0 = time.time()
    for step in range(start_step, args.steps):
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        for _ in range(args.grad_accum):
            try:
                x, y = next(data_stream)
            except StopIteration:
                dataloader = get_omega_v1_dataloader(batch_size=args.batch_size, block_size=args.block_size)
                data_stream = iter(dataloader)
                x, y = next(data_stream)
                
            x, y = x.to(device), y.to(device)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), ignore_index=-100)
                loss = loss / args.grad_accum
                
            scaler.scale(loss).backward()
            accum_loss += loss.item() * args.grad_accum
            
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        
        if step % 10 == 0:
            dt = time.time() - t0
            t0 = time.time()
            # Since accum_loss is the SUM of (CE_loss / 8) * 8 across 8 microbatches,
            # accum_loss is 8 * CE_loss. We divide by grad_accum here so the printed loss is accurate.
            true_ce_loss = accum_loss / args.grad_accum
            print(f"Omega v1 Step {step:6d} | Loss: {true_ce_loss:.4f} | Time/10-steps: {dt:.2f}s")
            
        if step > 0 and step % 2500 == 0:
            print(f"[System] Saving Checkpoint at step {step}...")
            torch.save({
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'sched': sched.state_dict(),
                'scaler': scaler.state_dict(),
                'step': step
            }, args.ckpt)

if __name__ == "__main__":
    train_omega_v1()
