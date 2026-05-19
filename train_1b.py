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
from data_1b import get_1b_dataloader

class NaNDetectedException(Exception):
    """Custom exception raised when NaN or Inf values are detected in the loss."""
    pass

def get_gpu_memory():
    """Queries nvidia-smi for active VRAM utilization metrics."""
    try:
        cmd = "nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        mem, temp = output.split(',')
        return f"{mem.strip()}MB | Temp: {temp.strip()}C"
    except:
        return "N/A"

def train_1b_cluster():
    device = torch.device('cuda')
    assert torch.cuda.is_available(), "RTX 3090 Ti CUDA Environment Not Found."
    
    # -------------------------------------------------------------------------
    # 1. SCALE CONFIGURATION TO 1 BILLION PARAMETERS
    # -------------------------------------------------------------------------
    config = HGDMConfig(
        d_model=2048,        # Width
        n_layers=18,         # Depth
        n_heads=32,          # Heads (32 * 64 = 2048)
        d_k=64,              # Triton constraint
        d_v=64,              # Triton constraint
        d_ff=5460,           # SwiGLU scale (~ 8/3 * d_model)
        vocab_size=256
    )
    
    print("================================================================")
    print("LAUNCHING SCALE EXECUTION: 1.0B RECURRENT PARADIGM SPRINT")
    print("================================================================")
    print("[Dataset] Mixture Proportions: 60% FineWeb-Edu, 25% English Wikipedia, 15% Clean Code")
    
    model = HGDMUltimate(config).to(device)
    # Ensure correct mode is propagated to submodules
    model.train() 
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Model] Natively compiled 1B Target Architecture.")
    print(f"[Model] Total Parameter Count: {param_count / 1e9:.3f} Billion")
    print(f"[Memory] Baseline Allocated VRAM: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    # -------------------------------------------------------------------------
    # 2. OPTIMIZER & PIPELINE SETUP
    # -------------------------------------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    
    # Block size 2048, batch size 2
    block_size = 2048
    batch_size = 2
    grad_accum_steps = 16  # Effective Batch Size = 2 * 16 * 2048 = 65,536 tokens per update
    
    dataloader = get_1b_dataloader(block_size=block_size, batch_size=batch_size)
    data_stream = iter(dataloader)
    
    # -------------------------------------------------------------------------
    # 3. AUTO-RESUME CHECKPOINT & JSONL LOG LOADING
    # -------------------------------------------------------------------------
    checkpoint_path = "hgdm_1b_latest.pt"
    log_jsonl_path = "train_1b_logs.jsonl"
    
    start_step = 0
    tokens_trained = 0
    
    if os.path.exists(checkpoint_path):
        print(f"[System] Found existing checkpoint at {checkpoint_path}. Resuming training...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
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
            
            # Rewrite clean history
            with open(log_jsonl_path, "w") as f:
                for entry in valid_entries:
                    f.write(json.dumps(entry) + "\n")
            print(f"[System] Pruned JSONL log file to step {start_step} ({len(valid_entries)} history logs retained).")
        except Exception as e:
            print(f"[System] Error reading/pruning JSONL logs: {e}. Resetting log file.")
            
    # Fast-forward streaming data loader using itertools.islice (extremely fast C-level iterator skip)
    if start_step > 0:
        print(f"[Dataset] Fast-forwarding streaming data loader to step {start_step}...")
        t_ff_start = time.time()
        batches_to_skip = start_step * grad_accum_steps
        data_stream = itertools.islice(data_stream, batches_to_skip, None)
        print(f"[Dataset] Fast-forward complete in {time.time() - t_ff_start:.2f}s.")
        
    print("[Optimizer] Initializing standard AdamW state vectors on-device...")
    t_start = time.time()
    
    # Helper to save checkpoint securely
    def save_checkpoint(current_step, current_tokens):
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'step': current_step,
            'tokens_trained': current_tokens
        }
        # Save to temporary file first, then replace atomic to prevent corruption on crash
        tmp_path = checkpoint_path + ".tmp"
        torch.save(state, tmp_path)
        if os.path.exists(tmp_path):
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            os.rename(tmp_path, checkpoint_path)

    # -------------------------------------------------------------------------
    # 4. TRAINING LOOP STEP
    # -------------------------------------------------------------------------
    step = start_step
    max_steps = 100000
    
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
                    # Divide loss by grad_accum_steps to average gradients correctly
                    loss = F.cross_entropy(logits.view(-1, 256), y.view(-1)) / grad_accum_steps
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    raise NaNDetectedException(f"NaN or Inf loss detected at Step {step} during accum step {accum_step}!")
                    
                # Accumulate gradients
                loss.backward()
                accum_loss += loss.item() * grad_accum_steps
            
            # Anchor gradients to prevent scale explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            step_time = time.time() - t_step_start
            tokens_trained += batch_size * block_size * grad_accum_steps
            bpb = accum_loss / math.log(2)
            
            # Log step metrics to append-only JSONL file (O(1) I/O write)
            log_entry = {
                "step": step,
                "loss": accum_loss,
                "bpb": bpb,
                "vram": get_gpu_memory(),
                "time": step_time,
                "tokens_trained": tokens_trained
            }
            with open(log_jsonl_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
            # Print frequency control:
            # - First 100 steps: Print every 25 steps (0, 25, 50, 75, 100)
            # - After 100 steps: Print every 100 steps
            is_print_step = (step <= 100 and step % 25 == 0) or (step > 100 and step % 100 == 0)
            if is_print_step:
                print(f"Step {step:5d} | Train Loss: {accum_loss:.4f} | BPB: {bpb:.4f} | Tokens Trained: {tokens_trained:,} | VRAM: {get_gpu_memory()} | Time: {step_time:.2f}s")
            
            # Save checkpoint every 100 steps, overwriting the existing checkpoint file
            if step > 0 and step % 100 == 0:
                save_checkpoint(step, tokens_trained)
                print(f"[System] Checkpoint saved successfully (overwritten) at step {step}.")
                
            step += 1
            
            # Thermal check to protect the local environment (cool down if >= 95C)
            if step % 5 == 0:
                try:
                    cmd = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits"
                    output = subprocess.check_output(cmd, shell=True).decode().strip()
                    gpu_temp = int(output)
                    if gpu_temp >= 95:
                        print(f"\n[THERMAL CONTROL] GPU temperature hit {gpu_temp}C (>= 95C). Sleeping for 180 seconds to cool down...")
                        time.sleep(180)
                except:
                    pass
            
    except NaNDetectedException as e:
        print(f"\n[CRITICAL ERROR] {e} Stopping training immediately. Checkpoint was NOT saved/updated to prevent weight corruption.")
        sys.exit(1)
        
    except (KeyboardInterrupt, SystemExit):
        print(f"\n[System] Training interrupted/stopped. Saving current state to {checkpoint_path}...")
        save_checkpoint(step, tokens_trained)
        print("[System] Save complete. Exiting.")
        
    except Exception as e:
        print(f"\n[System] Unexpected error encountered: {e}. Saving state before crash...")
        save_checkpoint(step, tokens_trained)
        raise e

    print(f"Training run completed in {(time.time() - t_start)/3600:.2f} hours.")

if __name__ == "__main__":
    train_1b_cluster()
