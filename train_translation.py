import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import math
import sys
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

# Add current path for module resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hgdm_omega import OmegaGDM, OmegaConfig

# =============================================================================
# DATA PIPELINE: Source-Masked Causal Translator
# =============================================================================

class TranslationIterableDataset(IterableDataset):
    """
    Streams cfilt/iitb-english-hindi, formats into causal pairs:
    'EN: {english} \n HI: {hindi} \n'
    Pads to block_size using 0x00 and creates a binary loss mask.
    """
    def __init__(self, block_size=512, split="train"):
        super().__init__()
        self.block_size = block_size
        self.split = split
        print(f"[Dataset] Initializing cfilt/iitb-english-hindi streaming ({split} split)...")
        self.dataset = load_dataset("cfilt/iitb-english-hindi", split=split, streaming=True)

    def __iter__(self):
        data_iter = iter(self.dataset)
        
        while True:
            try:
                item = next(data_iter)
                pair = item['translation']
                en_text = pair['en'].strip()
                hi_text = pair['hi'].strip()
                
                # Format prompts
                prefix = "EN: "
                middle = "\nHI: "
                suffix = "\n"  # acts as <EOS>
                
                # Encode parts to bytes
                prefix_bytes = list(prefix.encode('utf-8', errors='ignore'))
                en_bytes = list(en_text.encode('utf-8', errors='ignore'))
                mid_bytes = list(middle.encode('utf-8', errors='ignore'))
                hi_bytes = list(hi_text.encode('utf-8', errors='ignore'))
                suf_bytes = list(suffix.encode('utf-8', errors='ignore'))
                
                # Construct input and mask
                # English context: prefix + en_bytes + mid_bytes
                source_len = len(prefix_bytes) + len(en_bytes) + len(mid_bytes)
                # Hindi target: hi_bytes + suf_bytes
                target_len = len(hi_bytes) + len(suf_bytes)
                
                total_len = source_len + target_len
                if total_len > self.block_size:
                    # Skip sentence if too long for 512-byte block
                    continue
                
                # Combine input bytes
                seq_bytes = prefix_bytes + en_bytes + mid_bytes + hi_bytes + suf_bytes
                
                # Build target-only loss mask: 0 for English context, 1 for Hindi target
                mask_list = [0] * source_len + [1] * target_len
                
                # Pad to block_size with 0x00
                padding_len = self.block_size - total_len
                seq_bytes = seq_bytes + [0] * padding_len
                mask_list = mask_list + [0] * padding_len
                
                yield torch.tensor(seq_bytes, dtype=torch.long), torch.tensor(mask_list, dtype=torch.float32)
                
            except StopIteration:
                # Re-initialize stream if exhausted
                self.dataset = load_dataset("cfilt/iitb-english-hindi", split=self.split, streaming=True)
                data_iter = iter(self.dataset)
                continue

def get_dataloader(block_size=512, batch_size=16, split="train"):
    dataset = TranslationIterableDataset(block_size=block_size, split=split)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

# =============================================================================
# TRAINING LOOP
# =============================================================================

def get_gpu_memory():
    try:
        import subprocess
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        return int(subprocess.check_output(cmd, shell=True).decode().strip())
    except:
        return 0

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=4e-4, help="Peak learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Training Device: {device}")

    # Model Config (39.5 Million parameters)
    config = OmegaConfig(
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
    
    model = OmegaGDM(config, force_sequential=False).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[Model]  Parameters: {params/1e6:.2f} Million")

    # DataLoader (IITB Parallel Corpus)
    dataloader = get_dataloader(block_size=512, batch_size=args.batch_size, split="train")
    data_iter = iter(dataloader)

    # Optimizer & Scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Linear Warmup + Cosine Annealing
    warmup_steps = 500
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps - warmup_steps, eta_min=args.lr / 10.0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])

    model.train()
    t_start = time.time()
    checkpoint_path = "hgdm_translation_latest.pt"

    print(f"\n{'Step':<5} | {'Loss':<8} | {'BPB':<6} | {'VRAM':<8} | {'StepTime':<8} | {'Elapsed'}")
    print("-" * 65)

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        t_step = time.time()

        for _ in range(args.grad_accum):
            try:
                batch, mask = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch, mask = next(data_iter)

            batch = batch.to(device)
            mask = mask.to(device)

            # Causal alignment: inputs = x, targets = shifted right by 1
            x = batch[:, :-1]
            y = batch[:, 1:]
            
            # Loss mask is aligned with the targets
            y_mask = mask[:, 1:]

            # Forward pass with automatic bfloat16 mixed precision
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(x)
                # Compute raw per-token cross entropy loss
                raw_loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction='none')
                # Apply mask (0 for English/padding, 1 for Hindi target)
                masked_loss = raw_loss * y_mask.reshape(-1)
                
                # Normalise loss by total target bytes present in the batch
                norm = y_mask.sum()
                if norm > 0:
                    loss = masked_loss.sum() / norm
                else:
                    loss = masked_loss.sum() * 0.0 # handle empty target batch boundary case safely

                loss_scaled = loss / args.grad_accum

            if torch.isnan(loss_scaled) or torch.isinf(loss_scaled):
                continue

            loss_scaled.backward()
            accum_loss += loss.item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        # Logging stats
        step_time = time.time() - t_step
        bpb = accum_loss / math.log(2)
        elapsed = (time.time() - t_start) / 60.0
        vram = get_gpu_memory()

        if step % 50 == 0 or step == args.steps - 1:
            print(f"{step:05d} | {accum_loss:<8.4f} | {bpb:<6.3f} | {vram:<4} MB  | {step_time:.2f}s     | {elapsed:.1f}min")
            sys.stdout.flush()

        # Save checkpoint periodically
        if step > 0 and (step % 1000 == 0 or step == args.steps - 1):
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': accum_loss,
            }, checkpoint_path)
            print(f"[System] Saved translation model checkpoint to {checkpoint_path}")
            sys.stdout.flush()

    print(f"\n[System] Training Complete! Total time: {(time.time() - t_start)/60:.2f} minutes")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
