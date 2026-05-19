import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import subprocess
import math
import json
import sys
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
    model.training = True 
    
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
    dataloader = get_1b_dataloader(block_size=block_size, batch_size=batch_size)
    data_stream = iter(dataloader)
    
    # -------------------------------------------------------------------------
    # 3. AUTO-RESUME CHECKPOINT LOADING
    # -------------------------------------------------------------------------
    checkpoint_path = "hgdm_1b_latest.pt"
    log_json_path = "train_1b_logs.json"
    
    start_step = 0
    tokens_trained = 0
    logs = []
    
    if os.path.exists(checkpoint_path):
        print(f"[System] Found existing checkpoint at {checkpoint_path}. Resuming training...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint['step']
            tokens_trained = checkpoint.get('tokens_trained', start_step * batch_size * block_size)
            print(f"[System] Resumed successfully from Step {start_step} | Tokens trained: {tokens_trained:,}")
        except Exception as e:
            print(f"[System] Error loading checkpoint: {e}. Starting from scratch.")
            start_step = 0
            tokens_trained = 0
            
    if os.path.exists(log_json_path):
        try:
            with open(log_json_path, "r") as f:
                logs = json.load(f)
            # Filter logs up to start_step to keep history clean on resume
            logs = [l for l in logs if l.get('step', 0) < start_step]
            print(f"[System] Loaded {len(logs)} historic log entries from JSON.")
        except Exception as e:
            print(f"[System] Error loading JSON logs: {e}. Resetting log file.")
            logs = []
            
    # Fast-forward streaming data loader to current step
    if start_step > 0:
        print(f"[Dataset] Fast-forwarding streaming data loader to step {start_step}...")
        t_ff_start = time.time()
        for ff_step in range(start_step):
            _ = next(data_stream)
            if ff_step > 0 and ff_step % 500 == 0:
                print(f"[Dataset] Fast-forwarded {ff_step}/{start_step} batches...")
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
            
            # Load next batch
            batch = next(data_stream).to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            
            t_step_start = time.time()
            
            # Forward pass wrapped under native bfloat16 autocast
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                raise NaNDetectedException(f"NaN or Inf loss detected at Step {step}!")
                
            # Backward pass with gradient checkpointing recomputation
            loss.backward()
            
            # Anchor gradients to prevent scale explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            step_time = time.time() - t_step_start
            tokens_trained += batch_size * block_size
            bpb = loss.item() / math.log(2)
            
            # Log step metrics to memory list and serialize to JSON
            log_entry = {
                "step": step,
                "loss": loss.item(),
                "bpb": bpb,
                "vram": get_gpu_memory(),
                "time": step_time,
                "tokens_trained": tokens_trained
            }
            logs.append(log_entry)
            
            # Write JSON log file
            with open(log_json_path, "w") as f:
                json.dump(logs, f, indent=4)
                
            # Print frequency control:
            # - First 100 steps: Print every 25 steps (0, 25, 50, 75, 100)
            # - After 100 steps: Print every 100 steps
            is_print_step = (step <= 100 and step % 25 == 0) or (step > 100 and step % 100 == 0)
            if is_print_step:
                print(f"Step {step:5d} | Train Loss: {loss.item():.4f} | BPB: {bpb:.4f} | Tokens Trained: {tokens_trained:,} | VRAM: {get_gpu_memory()} | Time: {step_time:.2f}s")
            
            # Save checkpoint every 100 steps, overwriting the existing checkpoint file
            if step > 0 and step % 100 == 0:
                save_checkpoint(step, tokens_trained)
                print(f"[System] Checkpoint saved successfully (overwritten) at step {step}.")
                
            step += 1
            
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
