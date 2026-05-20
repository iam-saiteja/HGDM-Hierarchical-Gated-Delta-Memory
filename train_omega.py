import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import subprocess
import math
import json
import sys
import itertools
from hgdm_omega import OmegaGDM, OmegaConfig
from data_1b import get_1b_dataloader

class NaNDetectedException(Exception):
    """Custom exception raised when NaN or Inf values are detected in the loss."""
    pass

# Global caches for GPU metrics to avoid hot-loop subprocess overhead
cached_gpu_mem = "N/A"
cached_temp = "N/A"

def update_gpu_metrics():
    """Queries nvidia-smi for active VRAM and temperature metrics to update cache."""
    global cached_gpu_mem, cached_temp
    try:
        cmd = "nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        mem, temp = output.split(',')
        
        # Subtract preoccupied memory (11,570 MB) as requested
        total_used = int(mem.strip())
        preoccupied = 11570
        net_used = max(0, total_used - preoccupied)
        
        cached_gpu_mem = f"{net_used}MB"
        cached_temp = f"{temp.strip()}C"
    except Exception:
        pass

def get_gpu_memory():
    """Returns the cached net nvidia-smi VRAM metric."""
    global cached_gpu_mem
    return cached_gpu_mem

def get_gpu_temp():
    """Returns the cached GPU temperature metric."""
    global cached_temp
    return cached_temp

def train_omega():
    device = torch.device('cuda')
    assert torch.cuda.is_available(), "CUDA Environment Not Found."
    
    # Initialize metric cache at startup
    update_gpu_metrics()
    
    # -------------------------------------------------------------------------
    # 0. DATASET SPLIT & STREAM PRE-START VERIFICATION
    # -------------------------------------------------------------------------
    print("[Dataset] Running dataset split pre-start verification...")
    from datasets import load_dataset
    try:
        fw_sample = next(iter(load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)))
        print(f"[Dataset] FineWeb-Edu train split verified successfully! Sample text len: {len(fw_sample.get('text', ''))}")
        
        wiki_sample = next(iter(load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)))
        print(f"[Dataset] Wikipedia train split verified successfully! Sample title: {wiki_sample.get('title', 'N/A')}")
        
        code_sample = next(iter(load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)))
        print(f"[Dataset] CodeParrot-clean train split verified successfully! Sample content len: {len(code_sample.get('content', ''))}")
        
        print("[Dataset] All streaming pipelines successfully connected and verified!")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Dataset verification failed: {e}")
        print("Please check your network connection, dataset repositories, and Hugging Face access.")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # 1. MODEL CONFIGURATION
    # -------------------------------------------------------------------------
    config = OmegaConfig(
        d_byte=256,
        catcher_layers=2,
        renderer_layers=2,
        d_model=512,
        core_layers=12,
        n_heads=8,
        d_k=64,
        d_v=64,
        d_ff=2048,
        decimation_rate=8, # W = 8
        max_position_embeddings=2048,
        vocab_size=256
    )
    
    print("================================================================")
    print("LAUNCHING OMEGAGDM 1000-STEP TRAINING SPRINT")
    print("================================================================")
    print("[Dataset] Mixture Proportions: 60% FineWeb-Edu, 25% English Wikipedia, 15% Clean Code")
    
    model = OmegaGDM(config, force_sequential=False).to(device)
    model.train() 
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Model] Natively compiled OmegaGDM Target Architecture.")
    print(f"[Model] Total Parameter Count: {param_count / 1e6:.3f} Million")
    print(f"[Memory] Initial Net VRAM (nvidia-smi - 11,570MB): {get_gpu_memory()}")
    
    # -------------------------------------------------------------------------
    # 2. OPTIMIZER & PIPELINE SETUP
    # -------------------------------------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    
    block_size = 2048
    batch_size = 2
    grad_accum_steps = 16  # Effective Batch Size = 2 * 16 * 2048 = 65,536 tokens per update
    max_steps = 1000
    
    # Cosine Annealing Learning Rate Scheduler for quality convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=1e-5)
    
    dataloader = get_1b_dataloader(block_size=block_size, batch_size=batch_size)
    data_stream = iter(dataloader)
    
    # -------------------------------------------------------------------------
    # 3. AUTO-RESUME CHECKPOINT & LOGS LOADING
    # -------------------------------------------------------------------------
    checkpoint_path = "omega_checkpoint.pt"
    log_jsonl_path = "train_omega_logs.jsonl"
    
    start_step = 0
    tokens_trained = 0
    
    if os.path.exists(checkpoint_path):
        print(f"[System] Found existing checkpoint at {checkpoint_path}. Resuming training...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_step = checkpoint['step']
            tokens_trained = checkpoint.get('tokens_trained', start_step * batch_size * block_size * grad_accum_steps)
            print(f"[System] Resumed successfully from Step {start_step} | Tokens trained: {tokens_trained:,}")
        except Exception as e:
            print(f"[System] Error loading checkpoint: {e}. Starting from scratch.")
            start_step = 0
            tokens_trained = 0
            
    # Prune JSONL logs up to the start_step to maintain clean history on resume
    if os.path.exists(log_jsonl_path):
        try:
            valid_entries = []
            with open(log_jsonl_path, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get('step', 0) < start_step:
                            valid_entries.append(entry)
            
            with open(log_jsonl_path, "w") as f:
                for entry in valid_entries:
                    f.write(json.dumps(entry) + "\n")
            print(f"[System] Pruned JSONL log file to step {start_step} ({len(valid_entries)} history logs retained).")
        except Exception as e:
            print(f"[System] Error reading/pruning JSONL logs: {e}. Resetting log file.")
            
    # Fast-forward streaming data loader
    if start_step > 0:
        print(f"[Dataset] Fast-forwarding streaming data loader to step {start_step}...")
        t_ff_start = time.time()
        batches_to_skip = start_step * grad_accum_steps
        data_stream = itertools.islice(data_stream, batches_to_skip, None)
        print(f"[Dataset] Fast-forward complete in {time.time() - t_ff_start:.2f}s.")
        
    print("[Optimizer] Initializing standard AdamW state vectors on-device...")
    t_start = time.time()
    
    # -------------------------------------------------------------------------
    # 4. MONITORING AND LOG BUFFER CONTROLS
    # -------------------------------------------------------------------------
    log_buffer = []
    
    def flush_logs():
        nonlocal log_buffer
        if log_buffer:
            try:
                with open(log_jsonl_path, "a") as f:
                    for entry in log_buffer:
                        f.write(json.dumps(entry) + "\n")
                log_buffer.clear()
            except Exception as e:
                print(f"[System] Error flushing logs: {e}")

    # Helper to save checkpoint securely
    def save_checkpoint(current_step, current_tokens):
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'step': current_step,
            'tokens_trained': current_tokens
        }
        tmp_path = checkpoint_path + ".tmp"
        torch.save(state, tmp_path)
        if os.path.exists(tmp_path):
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            os.rename(tmp_path, checkpoint_path)

    # -------------------------------------------------------------------------
    # 5. TRAINING LOOP
    # -------------------------------------------------------------------------
    step = start_step
    
    try:
        while step < max_steps:
            opt.zero_grad(set_to_none=True)
            accum_loss = 0.0
            
            t_step_start = time.time()
            
            # Gradient Accumulation Loop
            for accum_step in range(grad_accum_steps):
                batch = next(data_stream).to(device)
                x = batch[:, :-1]
                y = batch[:, 1:]
                
                # Forward pass wrapped under native bfloat16 autocast
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)) / grad_accum_steps
                
                if torch.isnan(loss) or torch.isinf(loss):
                    raise NaNDetectedException(f"NaN or Inf loss detected at Step {step} during accum step {accum_step}!")
                    
                loss.backward()
                accum_loss += loss.item() * grad_accum_steps
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            
            step_time = time.time() - t_step_start
            tokens_trained += batch_size * block_size * grad_accum_steps
            bpb = accum_loss / math.log(2)
            
            # Run background hardware query only at step/print boundaries (saves CPU latency)
            if step % 5 == 0:
                update_gpu_metrics()
                
            # Log metrics
            log_entry = {
                "step": step,
                "loss": accum_loss,
                "bpb": bpb,
                "vram": get_gpu_memory(),
                "temp": get_gpu_temp(),
                "time": step_time,
                "tokens_trained": tokens_trained
            }
            log_buffer.append(log_entry)
            
            # Print status every 5 steps
            if step % 5 == 0 or step == start_step:
                lr = opt.param_groups[0]['lr']
                elapsed_total = time.time() - t_start
                print(f"Step {step:04d} | Loss: {accum_loss:.4f} | BPB: {bpb:.4f} | "
                      f"Net VRAM: {get_gpu_memory()} | Temp: {get_gpu_temp()} | "
                      f"LR: {lr:.2e} | StepTime: {step_time:.2f}s | Elapsed: {elapsed_total/60:.1f}min")
                sys.stdout.flush()
                
            # Save checkpoint and flush logs every 50 steps
            if step > 0 and step % 50 == 0:
                save_checkpoint(step, tokens_trained)
                flush_logs()
                print(f"[System] Checkpoint saved successfully at step {step} | Logs flushed.")
                sys.stdout.flush()
                
            step += 1
            
    except (KeyboardInterrupt, SystemExit):
        print("\n[System] Training interrupted. Saving checkpoint...")
        save_checkpoint(step, tokens_trained)
        flush_logs()
        print("[System] Checkpoint and logs saved.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Training crashed: {e}")
        save_checkpoint(step, tokens_trained)
        flush_logs()
        raise e
        
    # Flush any remaining logs on clean completion
    flush_logs()
    print("================================================================")
    print("TRAINING SPRINT COMPLETE.")
    print("================================================================")

if __name__ == "__main__":
    train_omega()
