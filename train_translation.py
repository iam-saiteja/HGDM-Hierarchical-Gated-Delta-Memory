import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import math
import argparse
import sys
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hgdm_omega import OmegaGDM, OmegaConfig

# =============================================================================
# DATA PIPELINE (IIT BOMBAY ENGLISH-HINDI STREAMING)
# =============================================================================

class TranslationDataset(IterableDataset):
    def __init__(self, split="train", block_size=512):
        super().__init__()
        self.block_size = block_size
        print(f"[Dataset] Loading '{split}' split of cfilt/iitb-english-hindi in streaming mode...")
        self.dataset = load_dataset("cfilt/iitb-english-hindi", split=split, streaming=True)

    def __iter__(self):
        buffer = []
        for item in self.dataset:
            trans = item.get('translation', {})
            en_text = trans.get('en', '')
            hi_text = trans.get('hi', '')
            if not en_text or not hi_text:
                continue

            # Format sequence causally: EN: {prompt} \n HI: {target} \n
            prompt = f"EN: {en_text}\nHI: "
            target = f"{hi_text}\n"

            prompt_bytes = prompt.encode('utf-8', errors='ignore')
            target_bytes = target.encode('utf-8', errors='ignore')
            
            full_seq = list(prompt_bytes) + list(target_bytes)
            if len(full_seq) >= self.block_size:
                # Skip sentence pairs exceeding block size limit to avoid truncation
                continue

            # Construct inputs and labels
            # input_ids: full sequence padded to block_size with 0x00
            input_ids = full_seq + [0] * (self.block_size - len(full_seq))
            
            # labels: shift inputs by 1, mask prompt and padding indices to -100
            labels = [-100] * self.block_size
            prompt_len = len(prompt_bytes)
            target_len = len(target_bytes)
            
            # input_ids[t] predicts labels[t] = input_ids[t+1]
            for t in range(prompt_len - 1, prompt_len + target_len - 1):
                labels[t] = input_ids[t+1]

            yield torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def get_dataloader(split="train", block_size=512, batch_size=16):
    dataset = TranslationDataset(split=split, block_size=block_size)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

# =============================================================================
# VALIDATION UTILITY
# =============================================================================

@torch.no_grad()
def run_eval(model, val_loader, device, num_samples=3):
    model.eval()
    print(f"\n{'─'*70}\n[Evaluation] Running validation translation samples...")
    
    samples_printed = 0
    # Create an iterator over val_loader and grab some samples
    for input_ids, labels in val_loader:
        for b in range(input_ids.shape[0]):
            if samples_printed >= num_samples:
                break
                
            # Locate prompt boundary
            # Find the position of 'HI: ' bytes inside input_ids[b]
            seq_bytes = input_ids[b].tolist()
            try:
                # Find the sequence 'HI: ' which is [72, 73, 58, 32]
                hi_marker = [72, 73, 58, 32]
                prompt_len = 0
                for i in range(len(seq_bytes) - len(hi_marker)):
                    if seq_bytes[i:i+len(hi_marker)] == hi_marker:
                        prompt_len = i + len(hi_marker)
                        break
            except ValueError:
                continue
                
            if prompt_len == 0:
                continue

            prompt_ids = seq_bytes[:prompt_len]
            # Get actual target bytes (excluding padding)
            target_ids = [x for x in seq_bytes[prompt_len:] if x != 0]
            
            prompt_str = bytes(prompt_ids).decode('utf-8', errors='replace')
            target_str = bytes(target_ids).decode('utf-8', errors='replace').strip()
            
            # Autoregressively generate translation
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            
            try:
                # Generate until newline '\n' (0x0a) or max_new_bytes
                generated = model.generate(prompt_tensor, max_new_bytes=150, temp=0.5)
                gen_ids = generated[0, len(prompt_ids):].cpu().tolist()
                
                # Truncate at newline to output clean single sentence
                if 10 in gen_ids:
                    gen_ids = gen_ids[:gen_ids.index(10)]
                elif 0 in gen_ids:
                    gen_ids = gen_ids[:gen_ids.index(0)]
                    
                decoded_str = bytes(gen_ids).decode('utf-8', errors='replace').strip()
            except Exception as e:
                decoded_str = f"[Generation Error: {e}]"
                
            print(f"\nSAMPLE {samples_printed+1}:")
            print(f"  SOURCE (EN): {prompt_str.replace('EN: ', '').replace('\\nHI: ', '').strip()}")
            print(f"  TARGET (HI): {target_str}")
            print(f"  MODEL  (HI): {decoded_str}")
            samples_printed += 1
            
        if samples_printed >= num_samples:
            break
            
    print(f"{'─'*70}\n")
    model.train()

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train HGDM Causal English-to-Hindi Translation Model")
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per step")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
    parser.add_argument("--lr", type=float, default=4e-4, help="Peak learning rate")
    parser.add_argument("--ckpt", default="hgdm_translation_latest.pt", help="Checkpoint save path")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[System] Training Device: {device}")

    # 1. Instantiate Model Configuration
    # Parameters target: ~39.5 Million parameters
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
        max_position_embeddings=2048,
        vocab_size=256,
        use_state_fusion=False
    )
    
    model = OmegaGDM(omega_cfg, force_sequential=False).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameter Count: {params/1e6:.2f} Million")

    # 2. Dataloaders
    train_loader = get_dataloader("train", block_size=512, batch_size=args.batch_size)
    val_loader = get_dataloader("validation", block_size=512, batch_size=args.batch_size)
    
    train_stream = iter(train_loader)

    # 3. Optimizer & Scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    warmup_steps = min(500, args.steps // 20)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps - warmup_steps, eta_min=args.lr / 10)
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])

    model.train()
    t_start = time.time()
    
    print(f"\n{'Step':<5} | {'Loss':<8} | {'Tokens Trained':<14} | {'StepTime':<8} | {'LR':<9} | {'Elapsed'}")
    print("─" * 70)
    
    tokens_trained = 0

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        t_step = time.time()
        
        try:
            for _ in range(args.grad_accum):
                input_ids, labels = next(train_stream)
                input_ids, labels = input_ids.to(device), labels.to(device)
                
                # Shift inputs and labels for causal language modeling
                x = input_ids[:, :-1]
                y = labels[:, :-1]
                
                with torch.amp.autocast(device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32):
                    logits, _ = model(x) # (B, T-1, vocab_size)
                    loss = F.cross_entropy(
                        logits.reshape(-1, 256), 
                        y.reshape(-1), 
                        ignore_index=-100
                    ) / args.grad_accum
                    
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                    
                loss.backward()
                accum_loss += loss.item() * args.grad_accum
                tokens_trained += (y != -100).sum().item()
                
        except StopIteration:
            train_stream = iter(train_loader)
            continue
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        
        step_time = time.time() - t_step
        elapsed_min = (time.time() - t_start) / 60
        current_lr = scheduler.get_last_lr()[0]

        if step % 25 == 0 or step == args.steps - 1:
            print(f"{step:05d} | {accum_loss:<8.4f} | {tokens_trained:<14,} | {step_time:.2f}s     | {current_lr:<9.2e} | {elapsed_min:.1f}min")
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
