import torch
import torch.nn.functional as F
import time
import os
import sys
import argparse
import signal
import json
import copy
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hgdm_omega import OmegaGDM, OmegaConfig
from slm.data_dpo import get_dpo_dataloader

def get_gpu_memory():
    try:
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        return int(subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        return -1

def get_gpu_temp():
    try:
        cmd = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits"
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip() + "C"
    except Exception:
        return "N/A"

def safe_save_checkpoint(state, ckpt_path):
    tmp_path = ckpt_path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, ckpt_path)

shutdown_requested = False

def handle_interrupt(signum, frame):
    global shutdown_requested
    print("\n[System] Shutdown signal received! Safely finishing DPO step...")
    shutdown_requested = True

def compute_logprobs(model, x, y):
    logits, _ = model(x)
    valid_mask = (y != -100)
    y_safe = torch.where(valid_mask, y, torch.zeros_like(y))
    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logprobs = log_probs.gather(dim=-1, index=y_safe.unsqueeze(-1)).squeeze(-1)
    return (per_token_logprobs * valid_mask).sum(dim=1)

def train_dpo_v1():
    parser = argparse.ArgumentParser(description="Train DPO Phase for Omega Model v1")
    parser.add_argument("--sft-ckpt", type=str, default="omega_v1_latest.pt", help="Path to the trained SFT checkpoint")
    parser.add_argument("--steps", type=int, default=15000, help="Total DPO training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=type(1e-5), default=1e-5, help="Peak learning rate for DPO")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO KL penalty coefficient")
    parser.add_argument("--block-size", type=int, default=2048, help="Sequence block length")
    parser.add_argument("--ckpt", type=str, default="omega_v1_dpo_latest.pt", help="Checkpoint save path")
    parser.add_argument("--logs", type=str, default="omega_v1_dpo_logs.jsonl", help="Logs JSONL save path")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    omega_cfg = OmegaConfig(
        d_byte=256, catcher_layers=2, renderer_layers=2,
        d_model=768, core_layers=12, n_heads=12,
        d_k=64, d_v=64, d_ff=3072,
        decimation_rate=8, max_position_embeddings=args.block_size,
        vocab_size=256, use_state_fusion=False
    )
    
    print("[Model] Initializing Reference Model and Policy Model...")
    model = OmegaGDM(omega_cfg, force_sequential=False).to(device)
    
    # Check if we are resuming DPO or starting fresh from SFT
    start_step = 0
    if os.path.exists(args.ckpt):
        print(f"[System] Resuming DPO from checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        start_step = ckpt['step']
    elif os.path.exists(args.sft_ckpt):
        print(f"[System] Starting fresh DPO from SFT checkpoint: {args.sft_ckpt}")
        ckpt = torch.load(args.sft_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
    else:
        print(f"[CRITICAL ERROR] SFT Checkpoint {args.sft_ckpt} not found!")
        sys.exit(1)
        
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    dataloader = get_dpo_dataloader(batch_size=args.batch_size, block_size=args.block_size)
    data_stream = iter(dataloader)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    if os.path.exists(args.ckpt):
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        scaler.load_state_dict(ckpt.get('scaler', scaler.state_dict()))

    print("=========================================")
    print("STARTING OMEGA V1 DPO RUN")
    print("=========================================")
    model.train()
    
    t0 = time.time()
    for step in range(start_step, args.steps):
        if shutdown_requested:
            print("[System] Graceful shutdown complete. You can resume safely later.")
            break

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_margin = 0.0
        nan_detected = False
        
        for _ in range(args.grad_accum):
            try:
                xc, yc, xr, yr = next(data_stream)
            except StopIteration:
                dataloader = get_dpo_dataloader(batch_size=args.batch_size, block_size=args.block_size)
                data_stream = iter(dataloader)
                xc, yc, xr, yr = next(data_stream)
                
            xc, yc, xr, yr = xc.to(device), yc.to(device), xr.to(device), yr.to(device)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.bfloat16):
                with torch.no_grad():
                    ref_logprobs_chosen = compute_logprobs(ref_model, xc, yc)
                    ref_logprobs_rejected = compute_logprobs(ref_model, xr, yr)
                    
                pi_logprobs_chosen = compute_logprobs(model, xc, yc)
                pi_logprobs_rejected = compute_logprobs(model, xr, yr)
                
                pi_logratios = pi_logprobs_chosen - pi_logprobs_rejected
                ref_logratios = ref_logprobs_chosen - ref_logprobs_rejected
                
                logits = pi_logratios - ref_logratios
                loss = -F.logsigmoid(args.beta * logits).mean()
                loss = loss / args.grad_accum
            
            if torch.isnan(loss) or torch.isinf(loss):
                nan_detected = True
                break
                
            scaler.scale(loss).backward()
            accum_loss += loss.item() * args.grad_accum
            accum_margin += logits.mean().item() / args.grad_accum
            
        if nan_detected:
            opt.zero_grad(set_to_none=True)
            continue
            
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            opt.zero_grad(set_to_none=True)
            continue

        scaler.step(opt)
        scaler.update()
        sched.step()
        
        if step % 10 == 0:
            dt = time.time() - t0
            t0 = time.time()
            gpu_mem = get_gpu_memory()
            gpu_temp = get_gpu_temp()
            print(f"DPO Step {step:6d} | Loss: {accum_loss:.4f} | Reward Margin: {accum_margin:.4f} | VRAM: {gpu_mem}MB | Temp: {gpu_temp} | Time/10-steps: {dt:.2f}s")
            
            log_data = {"step": step, "loss": round(accum_loss, 4), "margin": round(accum_margin, 4), "gpu_mem_mb": gpu_mem, "gpu_temp": gpu_temp}
            with open(args.logs, 'a') as f:
                f.write(json.dumps(log_data) + "\n")
            
        if (step > 0 and step % 500 == 0) or shutdown_requested:
            state = {
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'sched': sched.state_dict(),
                'scaler': scaler.state_dict(),
                'step': step
            }
            safe_save_checkpoint(state, args.ckpt)

    if not shutdown_requested:
        print(f"[System] DPO Training target reached! Saving final checkpoint at step {args.steps}...")
        state = {
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'scaler': scaler.state_dict(),
            'step': args.steps
        }
        safe_save_checkpoint(state, args.ckpt)
        print("[System] Final DPO checkpoint saved successfully. Omega v1 is fully aligned!")

if __name__ == "__main__":
    train_dpo_v1()
