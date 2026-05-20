import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import math
import json
import sys
import argparse

# Ensure we can import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hgdm_omega import OmegaConfig, OmegaGDM
from data_1b import get_1b_dataloader

def train_omega_100steps():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA is not available. This script must run on the GPU server.")
        return
        
    device = torch.device('cuda')
    print("================================================================")
    print("LAUNCHING OMEGAGDM 100-STEP DEMO TRAINING")
    print("================================================================")
    
    # 1. Configuration (Small model for validation, ~8M params)
    config = OmegaConfig(
        d_byte=128,
        catcher_layers=2,
        renderer_layers=2,
        d_model=256,
        core_layers=6,
        n_heads=4,
        d_k=64,
        d_v=64,
        d_ff=1024,
        decimation_rate=8, # W=8
        vocab_size=256,
        max_position_embeddings=2048
    )
    
    print("Initializing model...")
    model = OmegaGDM(config, force_sequential=False).to(device)
    model.train()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {params:,}")
    
    # 2. Optimizer & Data Setup
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    
    # Set block size and batch size
    block_size = 1024
    batch_size = 2
    grad_accum_steps = 4 # Small accum steps for fast demonstration
    max_steps = 100
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=1e-5)
    
    print("Connecting to data streams (FineWeb, Wikipedia, Code)...")
    dataloader = get_1b_dataloader(block_size=block_size, batch_size=batch_size)
    data_iter = iter(dataloader)
    
    # 3. Training Loop
    print("\nStarting training loop...")
    log_jsonl_path = "train_omega_100steps_logs.jsonl"
    logs = []
    
    # Remove previous demo log file if exists
    if os.path.exists(log_jsonl_path):
        os.remove(log_jsonl_path)
        
    t_start = time.time()
    
    for step in range(1, max_steps + 1):
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        t_step_start = time.perf_counter()
        
        for accum_step in range(grad_accum_steps):
            try:
                batch = next(data_iter).to(device)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter).to(device)
                
            x = batch[:, :-1]
            y = batch[:, 1:]
            
            # Forward pass with mixed precision autocast
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)) / grad_accum_steps
                
            loss.backward()
            accum_loss += loss.item() * grad_accum_steps
            
        # Optimization step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        
        step_time = time.perf_counter() - t_step_start
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        bpb = accum_loss / math.log(2)
        
        # Log to console
        if step == 1 or step % 10 == 0:
            print(f"Step {step:3d}/{max_steps} | Loss: {accum_loss:.4f} | BPB: {bpb:.4f} | GradNorm: {grad_norm.item():.4f} | VRAM: {vram_mb:.2f}MB | Time: {step_time:.2f}s")
            
        # Log entry
        log_entry = {
            "step": step,
            "loss": accum_loss,
            "bpb": bpb,
            "grad_norm": grad_norm.item(),
            "vram_mb": vram_mb,
            "time_sec": step_time
        }
        logs.append(log_entry)
        
    t_end = time.time()
    print("\n----------------------------------------------------------------")
    print(f"TRAINING COMPLETE IN {t_end - t_start:.2f} seconds!")
    print("----------------------------------------------------------------")
    
    # Save logs to disk
    with open(log_jsonl_path, "w") as f:
        for entry in logs:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved training logs to {log_jsonl_path}")
    
    # 4. Run Autoregressive Generation Test
    print("\nRunning Autoregressive Generation Test at step 100...")
    model.eval()
    
    # Simple prompt (represented as bytes for raw byte language modeling)
    prompt_text = "The Omega-GDM model is a new architecture designed specifically for"
    prompt_bytes = torch.tensor([[b for b in prompt_text.encode('utf-8')]], device=device, dtype=torch.long)
    
    print(f"Prompt: {prompt_text}")
    print("Generating continuation...")
    
    with torch.no_grad():
        generated_tokens = model.generate(prompt_bytes, max_new_bytes=64, temp=0.8)
        
    generated_bytes = generated_tokens[0].cpu().numpy().tolist()
    # Decode bytes back to text, ignoring errors
    decoded_text = bytes(generated_bytes).decode('utf-8', errors='replace')
    
    print(f"\nGenerated output:\n{decoded_text}")
    print("================================================================")

if __name__ == "__main__":
    train_omega_100steps()
