import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import math
import argparse
import sys
import subprocess
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hgdm_omega import OmegaGDM, OmegaConfig

# =============================================================================
# DATA PIPELINE (HUMAN DNA STREAMING)
# =============================================================================

class DNADataset(IterableDataset):
    def __init__(self, split="train", block_size=8192):
        super().__init__()
        self.block_size = block_size
        print(f"[Dataset] Loading '{split}' split of simecek/Human_DNA_v0 in streaming mode...")
        ds = load_dataset("simecek/Human_DNA_v0", split=split, streaming=True)
        if split == "train":
            ds = ds.shuffle(buffer_size=10000, seed=42)
        self.dataset = ds

    def __iter__(self):
        for item in self.dataset:
            seq = item.get('Seq', '')
            if not seq:
                continue
            seq = seq.upper()
            seq_bytes = list(seq.encode('ascii', errors='ignore'))
            
            # Ensure it fits block_size
            if len(seq_bytes) < self.block_size:
                seq_bytes = seq_bytes + [0] * (self.block_size - len(seq_bytes))
            else:
                seq_bytes = seq_bytes[:self.block_size]

            yield torch.tensor(seq_bytes, dtype=torch.long)

def get_dataloader(split="train", block_size=8192, batch_size=4):
    dataset = DNADataset(split=split, block_size=block_size)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

# =============================================================================
# VRAM & GPU UTILITIES
# =============================================================================

def get_gpu_memory():
    try:
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        return int(subprocess.check_output(cmd, shell=True).decode().strip())
    except Exception:
        return -1

def get_gpu_temp():
    try:
        cmd = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits"
        return subprocess.check_output(cmd, shell=True).decode().strip() + "C"
    except Exception:
        return "N/A"

# =============================================================================
# VALIDATION UTILITY
# =============================================================================

@torch.no_grad()
def run_eval(model, val_loader, device, num_samples=3):
    model.eval()
    print(f"\n{'─'*90}\n[Evaluation] Running validation DNA sequence evaluations...")
    
    val_stream = iter(val_loader)
    total_loss = 0.0
    eval_steps = 10
    
    # Calculate average loss on validation set
    for step in range(eval_steps):
        try:
            batch = next(val_stream)
        except StopIteration:
            val_stream = iter(val_loader)
            batch = next(val_stream)
            
        batch = batch.to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]
        
        with torch.amp.autocast(device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            total_loss += loss.item()
            
    avg_loss = total_loss / eval_steps
    avg_bpb = avg_loss / math.log(2)
    print(f"[Evaluation Result] Validation Average Loss: {avg_loss:.4f} | BPB: {avg_bpb:.4f}")
    
    # Run generative DNA test
    seed = "ATCGATCGATCGATCG"
    prompt_bytes = list(seed.encode('ascii'))
    prompt_tensor = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
    
    try:
        # Autoregressively generate 100 bases
        generated = model.generate(prompt_tensor, max_new_bytes=100, temp=0.5)
        gen_ids = generated[0, len(prompt_bytes):].cpu().tolist()
        
        # Keep only standard DNA bases for clean printing (A, C, G, T, N)
        dna_chars = []
        for x in gen_ids:
            char = chr(x).upper()
            if char in ('A', 'C', 'G', 'T', 'N'):
                dna_chars.append(char)
            else:
                dna_chars.append('.') # representation of unexpected character
                
        decoded_str = "".join(dna_chars)
    except Exception as e:
        decoded_str = f"[Generation Error: {e}]"
        
    print(f"\nGENERATIVE PROMPT (SEED): {seed}")
    print(f"MODEL DNA CONTINUATION   : {decoded_str}")
    print(f"{'─'*90}\n")
    model.train()
    return avg_loss

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train HGDM Causal DNA Sequence Model at 8,192 Context")
    parser.add_argument("--steps", type=int, default=5000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
    parser.add_argument("--lr", type=float, default=4e-4, help="Peak learning rate")
    parser.add_argument("--ckpt", default="hgdm_dna_latest.pt", help="Checkpoint save path")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[System] Training Device: {device}")

    # 1. Instantiate Model Configuration (context length 8,192)
    # Target size: ~39.5 Million parameters
    omega_cfg = OmegaConfig(
        d_byte=256,
        catcher_layers=1,
        renderer_layers=1,
        d_model=512,
        core_layers=8,
        n_heads=8,
        d_k=64,
        d_v=64,
        d_ff=2048,
        decimation_rate=8,
        max_position_embeddings=8192,
        vocab_size=256,
        use_state_fusion=False
    )
    
    model = OmegaGDM(omega_cfg, force_sequential=False).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameter Count: {params/1e6:.2f} Million")

    # 2. Dataloaders
    train_loader = get_dataloader("train", block_size=8192, batch_size=args.batch_size)
    val_loader = get_dataloader("test", block_size=8192, batch_size=args.batch_size)
    
    train_stream = iter(train_loader)

    # 3. Optimizer & Scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    warmup_steps = min(500, args.steps // 20)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps - warmup_steps, eta_min=args.lr / 10)
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])

    model.train()
    t_start = time.time()
    
    print(f"\n{'Step':<5} | {'Loss':<8} | {'BPB':<7} | {'Tokens Trained':<14} | {'VRAM':<8} | {'Temp':<5} | {'StepTime':<8} | {'LR':<9} | {'Elapsed'}")
    print("─" * 105)
    
    tokens_trained = 0

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        t_step = time.time()
        
        try:
            for _ in range(args.grad_accum):
                batch = next(train_stream)
                batch = batch.to(device)
                
                # Shift inputs and labels for causal language modeling
                x = batch[:, :-1]
                y = batch[:, 1:]
                
                with torch.amp.autocast(device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32):
                    logits, _ = model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, 256), 
                        y.reshape(-1)
                    ) / args.grad_accum
                    
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                    
                loss.backward()
                accum_loss += loss.item() * args.grad_accum
                tokens_trained += y.numel()
                
        except StopIteration:
            train_stream = iter(train_loader)
            continue
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        
        step_time = time.time() - t_step
        elapsed_min = (time.time() - t_start) / 60
        current_lr = scheduler.get_last_lr()[0]
        bpb = accum_loss / math.log(2)

        if step % 25 == 0 or step == args.steps - 1:
            vram_mb = get_gpu_memory()
            temp_c = get_gpu_temp()
            vram_str = f"{vram_mb}MB" if vram_mb >= 0 else "N/A"
            print(f"{step:05d} | {accum_loss:<8.4f} | {bpb:<7.4f} | {tokens_trained:<14,} | {vram_str:<8} | {temp_c:<5} | {step_time:.2f}s     | {current_lr:<9.2e} | {elapsed_min:.1f}min")
            sys.stdout.flush()

        # Regular Evaluation and Checkpoint Saving
        if (step > 0 and step % 500 == 0) or step == args.steps - 1:
            run_eval(model, val_loader, device, num_samples=3)
            
            # Save Checkpoint
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'tokens_trained': tokens_trained,
                'config': omega_cfg
            }
            torch.save(checkpoint, args.ckpt)
            print(f"[System] Checkpoint saved successfully at step {step} to '{args.ckpt}'")
            sys.stdout.flush()

    print("\n[System] Training Complete!")

if __name__ == "__main__":
    main()
