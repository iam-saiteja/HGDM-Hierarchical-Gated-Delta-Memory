import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import subprocess
import math
from hgdm_ultimate import HGDMUltimate, HGDMConfig
from data_1b import get_1b_dataloader

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
    model = HGDMUltimate(config).to(device)
    
    # Force native activation checkpointing flag on model layers
    model.training = True 
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Model] Natively compiled 1B Target Architecture.")
    print(f"[Model] Total Parameter Count: {param_count / 1e9:.3f} Billion")
    print(f"[Memory] Baseline Allocated VRAM: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    # -------------------------------------------------------------------------
    # 2. OPTIMIZER & PIPELINE SETUP
    # -------------------------------------------------------------------------
    # Native on-GPU AdamW
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    
    # Dynamic data loader streaming blocks of 2048 bytes
    # Batch size 2 fits cleanly under activation boundary limits
    dataloader = get_1b_dataloader(block_size=2048, batch_size=2)
    data_stream = iter(dataloader)
    
    # Simple Warmup profile
    print("[Optimizer] Initializing standard AdamW state vectors on-device...")
    t_start = time.time()
    
    # -------------------------------------------------------------------------
    # 3. TRAINING LOOP STEP
    # -------------------------------------------------------------------------
    step = 0
    max_steps = 100000  # Set target boundary step metrics
    
    while step < max_steps:
        opt.zero_grad(set_to_none=True)
        
        # Load next structured data chunk from stream
        batch = next(data_stream).to(device) # Shape: [B, T+1]
        x = batch[:, :-1]
        y = batch[:, 1:]
        
        t_step_start = time.time()
        try:
            # Forward pass wrapped under native bfloat16 autocast
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
            
            # Backward Pass triggers recomputation through checkpoint blocks
            loss.backward()
            
            # Anchor gradients to prevent scale explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            opt.step()
            
            step_time = time.time() - t_step_start
            bpb = loss.item() / math.log(2)
            print(f"Step {step:5d} | Train Loss: {loss.item():.4f} | BPB: {bpb:.4f} | VRAM: {get_gpu_memory()} | Time: {step_time:.2f}s")
                
            # Perform regular saving checkpoints
            if step > 0 and step % 1000 == 0:
                torch.save(model.state_dict(), f"hgdm_1b_step_{step}.pt")
                print(f"[System] Checkpoint saved successfully at step {step}.")
                
            step += 1
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[CRITICAL ERROR] Out of Memory encountered at step {step}!")
                print(f"Current VRAM Allocation Snapshot: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
                print(f"Peak VRAM Tracked: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
                raise e
            else:
                raise e

    print(f"Training run completed in {(time.time() - t_start)/3600:.2f} hours.")

if __name__ == "__main__":
    train_1b_cluster()
