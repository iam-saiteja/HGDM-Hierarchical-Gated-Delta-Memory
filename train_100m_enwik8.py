import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import subprocess
import math
import json
import sys
import urllib.request
import zipfile
import argparse

# Ensure local import paths work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hgdm_omega import OmegaGDM, OmegaConfig

def get_gpu_memory():
    """Returns nvidia-smi VRAM usage in MB, or torch memory reserved as a fallback."""
    try:
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        return int(subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        # Fallback to PyTorch reserved memory if nvidia-smi fails
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

def get_enwik8_data():
    """Downloads and returns Enwik8 data splits as long tensors."""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "enwik8.zip")
    data_path = os.path.join(data_dir, "enwik8")
    
    if not os.path.exists(data_path):
        if not os.path.exists(zip_path):
            print("[Data] Downloading enwik8 (100MB)...")
            url = "http://mattmahoney.net/dc/enwik8.zip"
            urllib.request.urlretrieve(url, zip_path)
        print("[Data] Extracting enwik8...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
    with open(data_path, 'rb') as f:
        data = f.read()
    
    n = len(data)
    train_data = torch.frombuffer(data[:int(n * 0.9)], dtype=torch.uint8).long()
    val_data = torch.frombuffer(data[int(n * 0.9):], dtype=torch.uint8).long()
    return train_data, val_data

@torch.no_grad()
def evaluate_model(model, val_data, seq_len=2048, batches=20, batch_size=4, device='cuda'):
    """Computes the average cross-entropy loss over random validation slices."""
    model.eval()
    total_loss = 0.0
    for _ in range(batches):
        ix = torch.randint(len(val_data) - seq_len - 1, (batch_size,))
        x = torch.stack([val_data[i:i+seq_len] for i in ix]).to(device)
        y = torch.stack([val_data[i+1:i+seq_len+1] for i in ix]).to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            total_loss += loss.item()
    model.train()
    return total_loss / batches

@torch.no_grad()
def run_generation_test(model, device, prompt_text="The capital of France is Paris. The capital of Germany is", max_new_bytes=100, temp=0.8):
    """Generates new bytes given a prompt while monitoring peak GPU memory usage."""
    model.eval()
    print("\n" + "="*70)
    print("RUNNING GENERATION TEST (VRAM MONITORING)")
    print("="*70)
    
    # Reset peak VRAM stats
    torch.cuda.reset_peak_memory_stats()
    vram_start_allocated = torch.cuda.memory_allocated(device) / (1024**2)
    vram_start_nvidia = get_gpu_memory()
    
    prompt_bytes = list(prompt_text.encode('utf-8', errors='ignore'))
    prompt_tensor = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
    
    t0 = time.time()
    generated = model.generate(prompt_tensor, max_new_bytes=max_new_bytes, temp=temp)
    t_gen = time.time() - t0
    
    new_bytes = generated[0, len(prompt_bytes):].tolist()
    decoded = bytes(new_bytes).decode('utf-8', errors='replace')
    
    vram_peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
    vram_end_nvidia = get_gpu_memory()
    
    print(f"PROMPT : {prompt_text!r}")
    print(f"OUTPUT : {decoded!r}")
    print(f"Time   : {t_gen:.2f}s | Speed: {max_new_bytes / t_gen:.1f} bytes/s")
    print(f"PyTorch VRAM: Start {vram_start_allocated:.1f} MB | Peak {vram_peak_allocated:.1f} MB")
    if vram_start_nvidia != -1 and vram_end_nvidia != -1:
        print(f"NVIDIA-SMI VRAM: Start {vram_start_nvidia} MB | End {vram_end_nvidia} MB")
    print("="*70 + "\n")
    model.train()

def main():
    parser = argparse.ArgumentParser(description="Train 100M OmegaGDM on Enwik8")
    parser.add_argument("--steps", type=int, default=5000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=type(4e-4), default=4e-4, help="Peak learning rate")
    parser.add_argument("--block-size", type=int, default=2048, help="Sequence block length")
    parser.add_argument("--ckpt", type=str, default="omega_100m_enwik8.pt", help="Checkpoint save path")
    parser.add_argument("--logs", type=str, default="train_100m_logs.jsonl", help="Logs JSONL save path")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Device selected: {device}")
    if device.type != 'cuda':
        print("[WARNING] CUDA is not available. Running on CPU instead!")

    # 1. Load Data
    print("[Dataset] Loading Enwik8 dataset...")
    train_data, val_data = get_enwik8_data()
    print(f"[Dataset] Train Size: {len(train_data):,} bytes | Val Size: {len(val_data):,} bytes")

    # 2. Model Config targeting ~126M Parameters
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
    
    print("[Model] Initializing OmegaGDM...")
    model = OmegaGDM(omega_cfg, force_sequential=False).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameter Count: {params/1e6:.2f}M")
    
    # 3. Setup optimizer & scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr/10)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    start_step = 0
    logs = []

    # Resume capability check
    if os.path.exists(args.ckpt):
        print(f"[System] Resuming from existing checkpoint: {args.ckpt}")
        try:
            checkpoint = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
            sched.load_state_dict(checkpoint['scheduler_state_dict'])
            start_step = checkpoint['step']
            print(f"[System] Resumed at step {start_step}")
        except Exception as e:
            print(f"[System] Failed to load checkpoint ({e}). Starting fresh.")

    # Logging header
    print(f"\n{'Step':<5} | {'Loss':<8} | {'Train BPB':<9} | {'Val BPB':<8} | {'VRAM':<8} | {'StepTime':<8} | {'LR':<8} | {'Elapsed'}")
    print("-" * 78)
    sys.stdout.flush()

    t_start = time.time()
    
    try:
        for step in range(start_step, args.steps):
            t_step = time.time()
            opt.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for _ in range(args.grad_accum):
                # Random index fetch
                ix = torch.randint(len(train_data) - args.block_size - 1, (args.batch_size,))
                x = torch.stack([train_data[i:i+args.block_size] for i in ix]).to(device)
                y = torch.stack([train_data[i+1:i+args.block_size+1] for i in ix]).to(device)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)) / args.grad_accum

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n[ERROR] NaN/Inf loss encountered at step {step}. Aborting training.")
                    sys.exit(1)

                scaler.scale(loss).backward()
                accum_loss += loss.item() * args.grad_accum

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            step_time = time.time() - t_step
            train_bpb = accum_loss / math.log(2)
            vram_mb = get_gpu_memory()
            
            val_bpb = None
            # Validation every 250 steps (and at step 0)
            if step % 250 == 0 or step == args.steps - 1:
                val_loss = evaluate_model(model, val_data, seq_len=args.block_size, batches=10, batch_size=args.batch_size, device=device)
                val_bpb = val_loss / math.log(2)
                run_generation_test(model, device)
            
            # Print status periodically
            if step % 25 == 0 or val_bpb is not None:
                elapsed = (time.time() - t_start) / 60
                val_str = f"{val_bpb:.4f}" if val_bpb is not None else "N/A"
                lr_curr = sched.get_last_lr()[0]
                temp = get_gpu_temp()
                temp_str = f" ({temp})" if temp != "N/A" else ""
                print(f"{step:04d}  | {accum_loss:<8.4f} | {train_bpb:<9.4f} | {val_str:<8} | {vram_mb:<4} MB  | {step_time:.2f}s    | {lr_curr:.2e} | {elapsed:.1f}min{temp_str}")
                sys.stdout.flush()

            # Save metrics
            log_entry = {
                "step": step,
                "loss": accum_loss,
                "train_bpb": train_bpb,
                "val_bpb": val_bpb,
                "vram_mb": vram_mb,
                "step_time": step_time,
                "lr": sched.get_last_lr()[0]
            }
            logs.append(log_entry)
            
            with open(args.logs, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Checkpoint override every 500 steps
            if step > 0 and step % 500 == 0 or step == args.steps - 1:
                print(f"[System] Saving checkpoint overriding existing file: {args.ckpt}")
                torch.save({
                    'step': step + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': sched.state_dict(),
                }, args.ckpt)

    except KeyboardInterrupt:
        print("\n[System] Training interrupted by user. Saving current state before exit...")
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': sched.state_dict(),
        }, args.ckpt)
        print("[System] Saved. Exiting.")
        sys.exit(0)

    print(f"\n[System] Training run completed successfully in {(time.time() - t_start)/60:.2f} minutes.")
    print(f"[System] Final model weights saved to: {args.ckpt}")

if __name__ == "__main__":
    main()
