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
# DATA PIPELINE (IIT BOMBAY ENGLISH-HINDI STREAMING)
# =============================================================================

class TranslationDataset(IterableDataset):
    def __init__(self, split="train", block_size=512):
        super().__init__()
        self.split = split
        self.block_size = block_size
        print(f"[Dataset] Loading '{split}' split of Helsinki-NLP/opus_books in streaming mode...")
        # Since it only has 'train' split, we load train and partition manually
        ds = load_dataset("Helsinki-NLP/opus_books", "en-es", split="train", streaming=True)
        # Shuffle using the same seed for both train/validation splits
        ds = ds.shuffle(buffer_size=10000, seed=42)
        self.dataset = ds

    def __iter__(self):
        count = 0
        for item in self.dataset:
            trans = item.get('translation', {})
            en_text = trans.get('en', '')
            es_text = trans.get('es', '')
            if not en_text or not es_text:
                continue

            # Partition: first 90,000 for training, rest for validation
            is_train_sample = count < 90000
            count += 1
            
            if self.split == "train" and not is_train_sample:
                break
            elif self.split == "validation" and is_train_sample:
                continue

            # Format sequence causally: EN: {prompt} \n ES: {target} \n
            prompt = f"EN: {en_text}\nES: "
            target = f"{es_text}\n"

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

def compute_bleu(reference, candidate):
    """
    Computes BLEU-4 score using NLTK if available, otherwise using a robust pure Python fallback.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens = reference.strip().split()
        cand_tokens = candidate.strip().split()
        if not ref_tokens or not cand_tokens:
            return 0.0
        # Method 1 smoothing adds a small epsilon to zero n-gram counts
        chen_cherry = SmoothingFunction()
        return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=chen_cherry.method1)
    except ImportError:
        # Fallback to pure-Python BLEU-4 with basic smoothing
        ref_tokens = reference.strip().split()
        cand_tokens = candidate.strip().split()
        if not ref_tokens or not cand_tokens:
            return 0.0
            
        import math
        from collections import Counter
        
        c = len(cand_tokens)
        r = len(ref_tokens)
        bp = 1.0 if c > r else (math.exp(1.0 - r / c) if c > 0 else 0.0)
        
        p_ns = []
        for n in range(1, 5):
            if len(cand_tokens) < n:
                p_ns.append(0.0)
                continue
            ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]
            cand_ngrams = [tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1)]
            
            ref_counts = Counter(ref_ngrams)
            cand_counts = Counter(cand_ngrams)
            
            clipped_hits = sum(min(count, ref_counts.get(ngram, 0)) for ngram, count in cand_counts.items())
            precision = clipped_hits / len(cand_ngrams)
            p_ns.append(precision)
            
        # Add-epsilon smoothing to prevent math.log(0)
        smoothed_p_ns = [p if p > 0 else 1e-5 for p in p_ns]
        s = sum(math.log(p) for p in smoothed_p_ns)
        return bp * math.exp(s / 4.0)

@torch.no_grad()
def run_eval(model, val_loader, device, num_samples=3, num_bleu_samples=50):
    model.eval()
    print(f"\n{'─'*70}\n[Evaluation] Running validation translation samples and calculating BLEU...")
    
    samples_printed = 0
    bleu_scores = []
    
    # Create an iterator over val_loader and grab some samples
    for input_ids, labels in val_loader:
        for b in range(input_ids.shape[0]):
            # Locate prompt boundary
            # Find the position of 'ES: ' bytes inside input_ids[b]
            seq_bytes = input_ids[b].tolist()
            try:
                # Find the sequence 'ES: ' which is [69, 83, 58, 32]
                es_marker = [69, 83, 58, 32]
                prompt_len = 0
                for i in range(len(seq_bytes) - len(es_marker)):
                    if seq_bytes[i:i+len(es_marker)] == es_marker:
                        prompt_len = i + len(es_marker)
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
                
            # Compute BLEU
            bleu = 0.0
            if target_str and not decoded_str.startswith("[Generation Error"):
                bleu = compute_bleu(target_str, decoded_str)
                bleu_scores.append(bleu)
                
            if samples_printed < num_samples:
                print(f"\nSAMPLE {samples_printed+1}:")
                # Remove EN: prefix and trailing newlines for clean printing
                src_clean = prompt_str.replace("EN: ", "").replace("\nES: ", "").strip()
                print(f"  SOURCE (EN): {src_clean}")
                print(f"  TARGET (ES): {target_str}")
                print(f"  MODEL  (ES): {decoded_str}")
                if target_str and not decoded_str.startswith("[Generation Error"):
                    print(f"  BLEU-4     : {bleu*100:.2f}%")
                samples_printed += 1
                
            if len(bleu_scores) >= num_bleu_samples:
                break
        if len(bleu_scores) >= num_bleu_samples:
            break
            
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    print(f"\n[Evaluation Result] Average BLEU-4 over {len(bleu_scores)} samples: {avg_bleu*100:.2f}%")
    print(f"{'─'*70}\n")
    model.train()

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
# MAIN TRAINING LOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train HGDM Causal English-to-Spanish Translation Model")
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
    
    start_step = 0
    tokens_trained = 0
    
    # Check if checkpoint exists to resume training
    if os.path.exists(args.ckpt):
        print(f"[System] Loading checkpoint '{args.ckpt}' to resume training...")
        checkpoint = torch.load(args.ckpt, map_location=device)
        if 'config' in checkpoint:
            omega_cfg = checkpoint['config']
            print(f"[System] Loaded model configuration directly from checkpoint.")
        model = OmegaGDM(omega_cfg, force_sequential=False).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint.get('step', 0) + 1
        tokens_trained = checkpoint.get('tokens_trained', 0)
        print(f"[System] Resumed successfully. Starting from step {start_step}")
    else:
        print("[System] No checkpoint found. Starting training from scratch.")
        model = OmegaGDM(omega_cfg, force_sequential=False).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        
    params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameter Count: {params/1e6:.2f} Million")

    # 2. Dataloaders
    train_loader = get_dataloader("train", block_size=512, batch_size=args.batch_size)
    val_loader = get_dataloader("validation", block_size=512, batch_size=args.batch_size)
    
    train_stream = iter(train_loader)

    # Fast-forward dataset stream to the correct step if resuming
    samples_to_skip = start_step * args.batch_size * args.grad_accum
    if samples_to_skip > 0:
        print(f"[System] Fast-forwarding dataset stream by {samples_to_skip:,} samples...")
        skipped = 0
        while skipped < samples_to_skip:
            try:
                next(train_stream)
                skipped += 1
            except StopIteration:
                train_stream = iter(train_loader)
                break

    # 4. Learning Rate Scheduler
    warmup_steps = min(500, args.steps // 20)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps - warmup_steps, eta_min=args.lr / 10)
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])

    # Fast-forward scheduler to the correct step if resuming
    if start_step > 0:
        print(f"[System] Fast-forwarding learning rate scheduler by {start_step} steps...")
        for _ in range(start_step):
            scheduler.step()

    model.train()
    t_start = time.time()
    
    print(f"\n{'Step':<5} | {'Loss':<8} | {'Tokens Trained':<14} | {'VRAM':<8} | {'Temp':<5} | {'StepTime':<8} | {'LR':<9} | {'Elapsed'}")
    print("─" * 90)

    for step in range(start_step, args.steps):
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
            vram_mb = get_gpu_memory()
            temp_c = get_gpu_temp()
            vram_str = f"{vram_mb}MB" if vram_mb >= 0 else "N/A"
            print(f"{step:05d} | {accum_loss:<8.4f} | {tokens_trained:<14,} | {vram_str:<8} | {temp_c:<5} | {step_time:.2f}s     | {current_lr:<9.2e} | {elapsed_min:.1f}min")
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
