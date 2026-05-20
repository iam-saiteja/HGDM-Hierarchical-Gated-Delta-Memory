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
from hgdm_ultimate import HGDMUltimate, HGDMConfig
from hgdm_omega import OmegaGDM, OmegaConfig
from data_1b import get_1b_dataloader

class NaNDetectedException(Exception):
    """Custom exception raised when NaN or Inf values are detected in the loss."""
    pass

# Global caches for GPU metrics to avoid hot-loop subprocess overhead
cached_gpu_mem = "N/A"
cached_temp = "N/A"
PREOCCUPIED_MEM = 11570  # MB preoccupied by friend

def update_gpu_metrics():
    """Queries nvidia-smi for active VRAM and temperature metrics to update cache."""
    global cached_gpu_mem, cached_temp
    try:
        cmd = "nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        mem, temp = output.split(',')
        
        # Subtract preoccupied memory (11,570 MB) as requested
        total_used = int(mem.strip())
        net_used = max(0, total_used - PREOCCUPIED_MEM)
        
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

def train_omega_comparison():
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
    # 1. SIDE-BY-SIDE MODEL CONFIGURATION (~25M-28M parameter scale)
    # -------------------------------------------------------------------------
    # Previous Monolithic HGDM
    config_hgdm = HGDMConfig(
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_k=64,
        d_v=64,
        d_ff=2048,
        max_position_embeddings=1024,
        vocab_size=256
    )
    
    # New OmegaGDM (Temporal Hourglass)
    config_omega = OmegaConfig(
        d_byte=256,
        catcher_layers=2,
        renderer_layers=2,
        d_model=512,
        core_layers=6,
        n_heads=8,
        d_k=64,
        d_v=64,
        d_ff=2048,
        decimation_rate=8, # W = 8
        max_position_embeddings=1024,
        vocab_size=256
    )
    
    print("================================================================")
    print("LAUNCHING HGDM VS OMEGAGDM COMPARISON TRAINING SPRINT")
    print("================================================================")
    print("[Dataset] Mixture Proportions: 60% FineWeb-Edu, 25% English Wikipedia, 15% Clean Code")
    
    model_hgdm = HGDMUltimate(config_hgdm, force_sequential=False).to(device)
    model_omega = OmegaGDM(config_omega, force_sequential=False).to(device)
    
    model_hgdm.train()
    model_omega.train()
    
    params_hgdm = sum(p.numel() for p in model_hgdm.parameters())
    params_omega = sum(p.numel() for p in model_omega.parameters())
    print(f"[Model] HGDM (Previous) Params: {params_hgdm / 1e6:.3f} Million")
    print(f"[Model] OmegaGDM (New) Params:   {params_omega / 1e6:.3f} Million")
    print(f"[Memory] Initial Net VRAM (nvidia-smi - 11,570MB): {get_gpu_memory()}")
    
    # -------------------------------------------------------------------------
    # 2. OPTIMIZERS & PIPELINE SETUP
    # -------------------------------------------------------------------------
    opt_hgdm = torch.optim.AdamW(model_hgdm.parameters(), lr=4e-4, weight_decay=0.01)
    opt_omega = torch.optim.AdamW(model_omega.parameters(), lr=4e-4, weight_decay=0.01)
    
    block_size = 1024
    batch_size = 2
    grad_accum_steps = 8  # Effective Batch Size = 2 * 8 * 1024 = 16,384 tokens per update
    max_steps = 1000
    
    # Cosine Annealing Learning Rate Schedulers
    sched_hgdm = torch.optim.lr_scheduler.CosineAnnealingLR(opt_hgdm, T_max=max_steps, eta_min=1e-5)
    sched_omega = torch.optim.lr_scheduler.CosineAnnealingLR(opt_omega, T_max=max_steps, eta_min=1e-5)
    
    dataloader = get_1b_dataloader(block_size=block_size, batch_size=batch_size)
    data_stream = iter(dataloader)
    
    # -------------------------------------------------------------------------
    # 3. AUTO-RESUME CHECKPOINT & LOGS LOADING
    # -------------------------------------------------------------------------
    checkpoint_path = "omega_comparison_checkpoint.pt"
    log_jsonl_path = "train_omega_comparison_logs.jsonl"
    
    start_step = 0
    tokens_trained = 0
    
    if os.path.exists(checkpoint_path):
        print(f"[System] Found existing checkpoint at {checkpoint_path}. Resuming training...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model_hgdm.load_state_dict(checkpoint['model_hgdm_state_dict'])
            model_omega.load_state_dict(checkpoint['model_omega_state_dict'])
            opt_hgdm.load_state_dict(checkpoint['opt_hgdm_state_dict'])
            opt_omega.load_state_dict(checkpoint['opt_omega_state_dict'])
            sched_hgdm.load_state_dict(checkpoint['sched_hgdm_state_dict'])
            sched_omega.load_state_dict(checkpoint['sched_omega_state_dict'])
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
            'model_hgdm_state_dict': model_hgdm.state_dict(),
            'model_omega_state_dict': model_omega.state_dict(),
            'opt_hgdm_state_dict': opt_hgdm.state_dict(),
            'opt_omega_state_dict': opt_omega.state_dict(),
            'sched_hgdm_state_dict': sched_hgdm.state_dict(),
            'sched_omega_state_dict': sched_omega.state_dict(),
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
    # 5. SIDE-BY-SIDE TRAINING LOOP
    # -------------------------------------------------------------------------
    step = start_step
    
    print("\n-------------------------------------------------------------------------------------------------")
    print(f"{'Step':<5} | {'HGDM Loss':<10} | {'Omega Loss':<10} | {'HGDM VRAM':<10} | {'Omega VRAM':<10} | {'Step Time':<10} | {'Elapsed':<10}")
    print("-------------------------------------------------------------------------------------------------")
    sys.stdout.flush()
    
    try:
        while step < max_steps:
            # Zero grads
            opt_hgdm.zero_grad(set_to_none=True)
            opt_omega.zero_grad(set_to_none=True)
            
            loss_hgdm_accum = 0.0
            loss_omega_accum = 0.0
            
            t_step_start = time.time()
            
            # Gradient Accumulation Loop
            for accum_step in range(grad_accum_steps):
                batch = next(data_stream).to(device)
                x = batch[:, :-1]
                y = batch[:, 1:]
                
                # --- HGDM Forward/Backward ---
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits_hgdm, _ = model_hgdm(x)
                    loss_hgdm = F.cross_entropy(logits_hgdm.reshape(-1, 256), y.reshape(-1)) / grad_accum_steps
                
                if torch.isnan(loss_hgdm) or torch.isinf(loss_hgdm):
                    raise NaNDetectedException(f"NaN/Inf HGDM loss detected at Step {step} during accum step {accum_step}!")
                loss_hgdm.backward()
                loss_hgdm_accum += loss_hgdm.item() * grad_accum_steps
                
                # --- OmegaGDM Forward/Backward ---
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits_omega, _ = model_omega(x)
                    loss_omega = F.cross_entropy(logits_omega.reshape(-1, 256), y.reshape(-1)) / grad_accum_steps
                
                if torch.isnan(loss_omega) or torch.isinf(loss_omega):
                    raise NaNDetectedException(f"NaN/Inf Omega loss detected at Step {step} during accum step {accum_step}!")
                loss_omega.backward()
                loss_omega_accum += loss_omega.item() * grad_accum_steps
            
            # Anchor gradients to prevent scale explosions
            torch.nn.utils.clip_grad_norm_(model_hgdm.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_omega.parameters(), 1.0)
            
            # Measure net active training memory (during active backprop/optimizer step)
            if step % 5 == 0:
                update_gpu_metrics()
                net_vram_active = get_gpu_memory()
            else:
                net_vram_active = cached_gpu_mem
                
            opt_hgdm.step()
            opt_omega.step()
            
            sched_hgdm.step()
            sched_omega.step()
            
            step_time = time.time() - t_step_start
            tokens_trained += batch_size * block_size * grad_accum_steps
            
            # Idle net memory after optimizer step releases activation graphs
            if step % 5 == 0:
                update_gpu_metrics()
                net_vram_idle = get_gpu_memory()
            else:
                net_vram_idle = cached_gpu_mem
                
            # Log metrics
            log_entry = {
                "step": step,
                "loss_hgdm": loss_hgdm_accum,
                "loss_omega": loss_omega_accum,
                "vram_hgdm_active": net_vram_active,
                "vram_omega_idle": net_vram_idle,
                "temp": get_gpu_temp(),
                "time": step_time,
                "tokens_trained": tokens_trained
            }
            log_buffer.append(log_entry)
            
            # Print status every 5 steps
            if step % 5 == 0 or step == start_step:
                elapsed_total = time.time() - t_start
                print(f"{step:04d} | {loss_hgdm_accum:.4f}    | {loss_omega_accum:.4f}     | {net_vram_active:<10} | {net_vram_idle:<10} | {step_time:.2f}s      | {elapsed_total/60:.1f}min")
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
    print("COMPARISON TRAINING SPRINT COMPLETE.")
    print("================================================================")

if __name__ == "__main__":
    train_omega_comparison()
