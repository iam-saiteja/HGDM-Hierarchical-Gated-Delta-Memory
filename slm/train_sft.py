import torch
import torch.nn.functional as F
import time
import os
import sys
import argparse
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hgdm_omega import OmegaGDM, OmegaConfig
from slm.data_sft import get_sft_dataloader

def train_sft():
    parser = argparse.ArgumentParser(description="Train SFT OmegaGDM SLM")
    parser.add_argument("--dataset", type=str, default="nvidia/HelpSteer2", help="HuggingFace dataset name")
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=type(2e-4), default=2e-4, help="Peak learning rate for SFT")
    parser.add_argument("--block-size", type=int, default=2048, help="Sequence block length")
    parser.add_argument("--base-ckpt", type=str, default="", help="Path to base pre-trained model (e.g., omega_100m_enwik8.pt)")
    parser.add_argument("--ckpt", type=str, default="omega_slm_sft_latest.pt", help="Checkpoint save path")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Device selected: {device}")

    # 1. Model Config targeting ~120M Parameters (GPT-1 scale)
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
    
    print("[Model] Initializing OmegaGDM for SFT...")
    model = OmegaGDM(omega_cfg, force_sequential=False).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameter Count: {params/1e6:.2f}M")

    # Load base weights if starting fresh from a pre-trained model
    if args.base_ckpt and not os.path.exists(args.ckpt):
        if os.path.exists(args.base_ckpt):
            print(f"[System] Loading base weights from {args.base_ckpt}...")
            base_ckpt = torch.load(args.base_ckpt, map_location=device)
            # Handle different checkpoint formats (model vs model_state_dict)
            state_dict = base_ckpt.get('model_state_dict', base_ckpt.get('model', base_ckpt))
            model.load_state_dict(state_dict)
            print("[System] Base weights loaded successfully.")
        else:
            print(f"[WARNING] Base checkpoint {args.base_ckpt} not found! Starting from scratch.")

    # 2. Setup Data Loader
    print(f"[Dataset] Initializing SFT stream from: {args.dataset}")
    dataloader = get_sft_dataloader(
        dataset_name=args.dataset, 
        split="train", 
        batch_size=args.batch_size, 
        block_size=args.block_size
    )
    data_stream = iter(dataloader)

    # 3. Setup optimizer & scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    start_step = 0
    if os.path.exists(args.ckpt):
        print(f"[System] Resuming SFT training from checkpoint {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        start_step = ckpt['step']
        scaler.load_state_dict(ckpt['scaler'])

    print("=========================================")
    print("STARTING SUPERVISED FINE-TUNING (SFT)")
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
                # If dataset ends, re-init stream (though streaming dataset loops or ends)
                dataloader = get_sft_dataloader(args.dataset, split="train", batch_size=args.batch_size, block_size=args.block_size)
                data_stream = iter(dataloader)
                x, y = next(data_stream)
                
            x, y = x.to(device), y.to(device)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.bfloat16):
                logits, _ = model(x)
                # Key detail: ignore_index=-100 masks out the user prompt from the loss
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
            print(f"SFT Step {step:4d} | Loss: {accum_loss:.4f} | Time/10-steps: {dt:.2f}s")
            
        if step > 0 and step % 500 == 0:
            print(f"[System] Saving Checkpoint at step {step}...")
            torch.save({
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'sched': sched.state_dict(),
                'scaler': scaler.state_dict(),
                'step': step
            }, args.ckpt)

if __name__ == "__main__":
    train_sft()
