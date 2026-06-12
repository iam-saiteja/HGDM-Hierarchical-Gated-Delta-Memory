import torch
import torch.nn.functional as F
import time
import os
import sys
import argparse
import signal
import json
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hgdm_omega import OmegaGDM, OmegaConfig
from v1.data_omega_v1 import get_omega_v1_dataloader

def get_gpu_memory():
    """Returns nvidia-smi VRAM usage in MB, or torch memory reserved as a fallback."""
    try:
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        return int(subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        if torch.cuda.is_available():
            return int(torch.cuda.memory_reserved() / (1024**2))
        return -1

def get_gpu_temp():
    """Returns nvidia-smi GPU temperature, or N/A as fallback."""
    try:
        cmd = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits"
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip() + "C"
    except Exception:
        return "N/A"

def safe_save_checkpoint(state, ckpt_path):
    """Saves checkpoint to a temporary file first, then renames it, preventing corruption if interrupted."""
    tmp_path = ckpt_path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, ckpt_path)

# Global flag for graceful shutdown
shutdown_requested = False

def handle_interrupt(signum, frame):
    global shutdown_requested
    print("\n[System] Shutdown signal received! Finishing current step and safely saving checkpoint...")
    shutdown_requested = True

def train_omega_v1():
    parser = argparse.ArgumentParser(description="Train Omega Model Version 1")
    parser.add_argument("--steps", type=int, default=150000, help="Total training steps for 1 epoch of 4.5M samples")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=type(3e-4), default=3e-4, help="Peak learning rate")
    parser.add_argument("--block-size", type=int, default=2048, help="Sequence block length")
    parser.add_argument("--ckpt", type=str, default="omega_v1_latest.pt", help="Checkpoint save path")
    parser.add_argument("--logs", type=str, default="omega_v1_logs.jsonl", help="Logs JSONL save path")
    args = parser.parse_args()

    # Register interrupt handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Device selected: {device}")

    # Omega Model 1 Specification (~120M Parameters)
    omega_cfg = OmegaConfig(
        d_byte=256, catcher_layers=2, renderer_layers=2,
        d_model=768, core_layers=12, n_heads=12,
        d_k=64, d_v=64, d_ff=3072,
        decimation_rate=8, max_position_embeddings=args.block_size,
        vocab_size=256, use_state_fusion=False
    )
    
    print("[Model] Initializing Omega Model v1...")
    model = OmegaGDM(omega_cfg, force_sequential=False).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameter Count: {params/1e6:.2f}M")

    print("[Dataset] Initializing OpenHermes-2.5 & OpenOrca stream with Identity Injection...")
    dataloader = get_omega_v1_dataloader(batch_size=args.batch_size, block_size=args.block_size)
    data_stream = iter(dataloader)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    start_step = 0
    tokens_trained = 0
    if os.path.exists(args.ckpt):
        try:
            print(f"[System] Resuming from checkpoint {args.ckpt}")
            ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            opt.load_state_dict(ckpt['opt'])
            sched.load_state_dict(ckpt['sched'])
            start_step = ckpt['step']
            tokens_trained = ckpt.get('tokens_trained', start_step * args.batch_size * args.grad_accum * args.block_size)
            scaler.load_state_dict(ckpt['scaler'])
            print(f"[System] Resumed successfully at step {start_step}.")
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to load checkpoint: {e}. Starting fresh.")

    print("=========================================")
    print(f"STARTING OMEGA V1 RUN (Target: {args.steps} steps)")
    print("=========================================")
    model.train()
    
    t0 = time.time()
    for step in range(start_step, args.steps):
        if shutdown_requested:
            print("[System] Executing graceful shutdown...")
            break

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        nan_detected = False
        
        for _ in range(args.grad_accum):
            try:
                x, y = next(data_stream)
            except StopIteration:
                dataloader = get_omega_v1_dataloader(batch_size=args.batch_size, block_size=args.block_size)
                data_stream = iter(dataloader)
                x, y = next(data_stream)
                
            x, y = x.to(device), y.to(device)
            tokens_trained += x.numel()
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), ignore_index=-100)
                loss = loss / args.grad_accum
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[WARNING] NaN/Inf loss detected at step {step}! Skipping this micro-batch to protect training stability.")
                nan_detected = True
                break
                
            scaler.scale(loss).backward()
            accum_loss += loss.item() * args.grad_accum

        if nan_detected:
            opt.zero_grad(set_to_none=True)
            continue
            
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"\n[WARNING] NaN/Inf grad_norm detected at step {step}! Skipping optimizer step.")
            opt.zero_grad(set_to_none=True)
            continue

        scaler.step(opt)
        scaler.update()
        sched.step()
        
        # Logging
        if step % 10 == 0:
            dt = time.time() - t0
            t0 = time.time()
            true_ce_loss = accum_loss / args.grad_accum
            bpb = true_ce_loss / 0.693147  # Convert nats to bits-per-byte (ln(2))
            
            gpu_mem = get_gpu_memory()
            gpu_temp = get_gpu_temp()
            
            print(f"Omega v1 Step {step:6d} | Loss: {true_ce_loss:.4f} | BPB: {bpb:.4f} | Tokens: {tokens_trained/1e6:.2f}M | VRAM: {gpu_mem}MB | Temp: {gpu_temp} | Time/10-steps: {dt:.2f}s")
            
            log_data = {
                "step": step,
                "loss": round(true_ce_loss, 4),
                "bpb": round(bpb, 4),
                "tokens_trained": tokens_trained,
                "gpu_mem_mb": gpu_mem,
                "gpu_temp": gpu_temp,
                "time_taken": round(dt, 2)
            }
            with open(args.logs, 'a') as f:
                f.write(json.dumps(log_data) + "\n")
            
        if (step > 0 and step % 2500 == 0) or shutdown_requested:
            print(f"[System] Saving Checkpoint at step {step}...")
            state = {
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'sched': sched.state_dict(),
                'scaler': scaler.state_dict(),
                'step': step,
                'tokens_trained': tokens_trained
            }
            safe_save_checkpoint(state, args.ckpt)

    if shutdown_requested:
        print("[System] Graceful shutdown complete. You can resume safely later.")

if __name__ == "__main__":
    train_omega_v1()
