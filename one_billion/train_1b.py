import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import subprocess
import math
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
        cached_gpu_mem = f"{mem.strip()}MB"
        cached_temp = f"{temp.strip()}C"
    except:
        pass

def get_gpu_memory():
    """Returns the cached nvidia-smi VRAM metric."""
    global cached_gpu_mem
    return cached_gpu_mem

def train_1b_cluster():
    device = torch.device('cuda')
    assert torch.cuda.is_available(), "RTX 3090 Ti CUDA Environment Not Found."
    
    # Initialize metric cache at startup
    update_gpu_metrics()
    
    # -------------------------------------------------------------------------
    # 0. DATASET SPLIT & STREAM PRE-START VERIFICATION
    # -------------------------------------------------------------------------
    print("[Dataset] Running dataset split pre-start verification...")
    from datasets import load_dataset
    try:
        # Load a single sample from each stream to verify connection and split configuration
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
    # 1. SCALE CONFIGURATION TO 1 BILLION PARAMETERS
    # -------------------------------------------------------------------------
    config = OmegaConfig(
        d_byte=256,
        catcher_layers=2,
        renderer_layers=2,
        d_model=2048,        # Width
        core_layers=18,      # Depth
        n_heads=32,          # Heads (32 * 64 = 2048)
        d_k=64,              # Triton constraint
        d_v=64,              # Triton constraint
        d_ff=5460,           # SwiGLU scale (~ 8/3 * d_model)
        decimation_rate=8,
        max_position_embeddings=65536,
        vocab_size=256,
        use_state_fusion=False
    )
    
    print("================================================================")
    print("LAUNCHING SCALE EXECUTION: 1.0B RECURRENT PARADIGM SPRINT")
    print("================================================================")
    print("[Dataset] Mixture Proportions: 60% FineWeb-Edu, 25% English Wikipedia, 15% Clean Code")
    
    model = OmegaGDM(config).to(device)
    model.train() 
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Model] Natively compiled 1B Target Architecture.")
    print(f"[Model] Total Parameter Count: {param_count / 1e9:.3f} Billion")
    print(f"[Memory] Baseline Allocated VRAM (nvidia-smi): {get_gpu_memory()}")
    
    # -------------------------------------------------------------------------
    # 2. OPTIMIZER & PIPELINE SETUP
    # -------------------------------------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    
    # Block size 2048, batch size 2
    block_size = 2048
    batch_size = 2
    grad_accum_steps = 16  # Effective Batch Size = 2 * 16 * 2048 = 65,536 tokens per update
    max_steps = 100000
    
    # Cosine Annealing Learning Rate Scheduler for quality convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=1e-5)
    
    dataloader = get_1b_dataloader(block_size=block_size, batch_size=batch_size)
    data_stream = iter(dataloader)
    
    # -------------------------------------------------------------------------
    # 3. AUTO-RESUME CHECKPOINT & LOGS LOADING
    # -------------------------------------------------------------------------
    checkpoint_path = "hgdm_1b_latest.pt"
    log_jsonl_path = "train_1b_logs.jsonl"
    
    start_step = 0
    tokens_trained = 0
    
    if os.path.exists(checkpoint_path):
        print(f"[System] Found existing checkpoint at {checkpoint_path}. Resuming training...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    
    # -------------------------------------------------------------------------
    # 4. MONITORING AND LOG BUFFER CONTROLS
    # -------------------------------------------------------------------------
    log_buffer = []
    
    def flush_logs():
        """Flushes the buffered JSONL logs to disk in a single batch I/O write."""
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
        # Save to temporary file first, then replace atomic to prevent corruption on crash
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
                    # Divide loss by grad_accum_steps to average gradients correctly
                    loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)) / grad_accum_steps
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    raise NaNDetectedException(f"NaN or Inf loss detected at Step {step} during accum step {accum_step}!")
                    
                # Accumulate gradients
                loss.backward()
                accum_loss += loss.item() * grad_accum_steps
            
            # Anchor gradients to prevent scale explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            
            step_time = time.time() - t_step_start
            tokens_trained += batch_size * block_size * grad_accum_steps
            bpb = accum_loss / math.log(2)
            
            # Buffer log entries in memory (reduces I/O overhead)
            log_entry = {
                "step": step,
                "loss": accum_loss,
                "bpb": bpb,
                "vram": get_gpu_memory(),
                "time": step_time,
                "tokens_trained": tokens_trained
            }
            log_buffer.append(log_entry)
            
            # Flush log buffer to file every 10 steps
            if len(log_buffer) >= 10:
                flush_logs()
                
            # Print frequency control:
            # - First 100 steps: Print every 25 steps (0, 25, 50, 75, 100)
            # - After 100 steps: Print every 100 steps
            is_print_step = (step <= 100 and step % 25 == 0) or (step > 100 and step % 100 == 0)
            
            # Update nvidia-smi cache only on print or checkpoint steps (prevents hot-loop bottlenecks)
            if is_print_step or (step > 0 and step % 100 == 0):
                update_gpu_metrics()
                
            if is_print_step:
                print(f"Step {step:5d} | Train Loss: {accum_loss:.4f} | BPB: {bpb:.4f} | Tokens Trained: {tokens_trained:,} | VRAM: {get_gpu_memory()} (Temp: {cached_temp}) | Time: {step_time:.2f}s")
            
            # Save checkpoint and perform thermal safety check every 100 steps
            if step > 0 and step % 100 == 0:
                save_checkpoint(step, tokens_trained)
                flush_logs()
                
                # Check thermals using cached temperature
                try:
                    gpu_temp = int(cached_temp.replace('C', ''))
                    if gpu_temp >= 95:
                        print(f"\n[THERMAL CONTROL] GPU temperature hit {gpu_temp}C (>= 95C). Sleeping for 180 seconds to cool down...")
                        time.sleep(180)
                        # Re-update metrics after sleep
                        update_gpu_metrics()
                except:
                    pass
                    
                print(f"[System] Checkpoint and logs saved successfully (overwritten) at step {step}.")
                
            step += 1
            
    except NaNDetectedException as e:
        flush_logs()
        print(f"\n[CRITICAL ERROR] {e} Stopping training immediately. Checkpoint was NOT saved/updated to prevent weight corruption.")
        sys.exit(1)
        
    except (KeyboardInterrupt, SystemExit):
        print(f"\n[System] Training interrupted/stopped. Saving current state to {checkpoint_path}...")
        flush_logs()
        save_checkpoint(step, tokens_trained)
        print("[System] Save complete. Exiting.")
        
    except Exception as e:
        print(f"\n[System] Unexpected error encountered: {e}. Saving state before crash...")
        flush_logs()
        save_checkpoint(step, tokens_trained)
        raise e

    # Flush final logs before script termination
    flush_logs()
    print(f"Training run completed in {(time.time() - t_start)/3600:.2f} hours.")

if __name__ == "__main__":
    train_1b_cluster()
