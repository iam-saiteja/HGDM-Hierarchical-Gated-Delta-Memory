import os
import sys
import time
import math
import argparse
import json
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from typing import Optional, Tuple, List, Dict

# Ensure local imports work cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenized.config import TokenizedConfig
from tokenized.model_hgdm import TokenizedHGDM
from tokenized.model_transformer import MatchedTransformer
from tokenized.dataset import create_dataloader


def get_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def train_single_model(
    model_type: str,
    config: TokenizedConfig,
    dataset_name: str,
    tokenizer_name: str,
    batch_size: int = 4,
    accum_steps: int = 4,
    seq_len: int = 1024,
    lr: float = 5e-4,
    max_tokens: Optional[int] = None,
    save_dir: str = "checkpoints_tokenized",
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Instantiate Model
    print("=" * 70)
    print(f"STARTING TRAINING: {model_type.upper()}")
    print("=" * 70)
    
    if model_type.lower() == "hgdm":
        model = TokenizedHGDM(config).to(device)
    elif model_type.lower() == "transformer":
        model = MatchedTransformer(config).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    target_tokens = max_tokens if max_tokens is not None else int(param_count * config.chinchilla_ratio)
    
    print(f"Model: {model_type.upper()}")
    print(f"Trainable Parameters: {param_count:,}")
    print(f"Target Chinchilla Tokens: {target_tokens:,} ({config.chinchilla_ratio:.0f}x params)")
    print(f"Batch Size (effective): {batch_size * accum_steps * seq_len:,} tokens/step")
    print("-" * 70)
    
    # 2. Data Loader
    loader = create_dataloader(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        batch_size=batch_size,
        seq_len=seq_len,
        max_tokens=target_tokens,
    )
    
    # 3. Optimizer & Cosine Schedule
    tokens_per_step = batch_size * accum_steps * seq_len
    total_steps = max(10, math.ceil(target_tokens / tokens_per_step))
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=lr * 0.1
    )
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    
    model.train()
    tokens_processed = 0
    step = 0
    t0 = time.time()
    
    data_iter = iter(loader)
    history = []
    
    while tokens_processed < target_tokens and step < total_steps:
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        for _ in range(accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
                
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / accum_steps
                
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            accum_loss += loss.item() * accum_steps
            tokens_processed += x.numel()
            
        if scaler:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
        scheduler.step()
        step += 1
        
        # Logging
        if step % 20 == 0 or step == 1 or tokens_processed >= target_tokens:
            elapsed = time.time() - t0
            tok_per_sec = tokens_processed / max(1e-4, elapsed)
            ppl = math.exp(min(20.0, accum_loss))
            vram = get_vram_mb()
            cur_lr = opt.param_groups[0]['lr']
            
            print(
                f"Step {step:4d}/{total_steps} | "
                f"Tokens: {tokens_processed:>9,}/{target_tokens:,} ({tokens_processed/target_tokens*100:4.1f}%) | "
                f"Loss: {accum_loss:.4f} | PPL: {ppl:6.2f} | "
                f"Throughput: {tok_per_sec:7.0f} tok/s | VRAM: {vram:5.0f}MB | LR: {cur_lr:.2e}"
            )
            
            history.append({
                "step": step,
                "tokens": tokens_processed,
                "loss": accum_loss,
                "ppl": ppl,
                "tok_per_sec": tok_per_sec,
                "vram_mb": vram,
                "time_s": elapsed,
            })
            
    # Save checkpoint
    ckpt_path = os.path.join(save_dir, f"ckpt_{model_type}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "model_type": model_type,
        "tokens_processed": tokens_processed,
        "history": history,
    }, ckpt_path)
    print(f"\n[Saved Checkpoint] {ckpt_path}")
    
    return ckpt_path, history


def main():
    parser = argparse.ArgumentParser(description="Chinchilla-Optimal Tokenized Arena Trainer")
    parser.add_argument("--model", type=str, choices=["hgdm", "transformer", "both"], default="both")
    parser.add_argument("--tier", type=str, choices=["pilot", "standard", "paper"], default="pilot")
    parser.add_argument("--dataset", type=str, default="HuggingFaceTB/smoltalk")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_tokens", type=int, default=None, help="Override Chinchilla target tokens")
    parser.add_argument("--save_dir", type=str, default="checkpoints_tokenized")
    args = parser.parse_args()
    
    # 1. Resolve Config
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = tok.vocab_size
    
    config = TokenizedConfig.get_tier(args.tier, vocab_size=vocab_size)
    
    print("=" * 70)
    print("CHINCHILLA TOKENIZED ARENA: HGDM vs. TRANSFORMER")
    print("=" * 70)
    print(f"Scale Tier: {args.tier.upper()}")
    print(f"Vocabulary: {vocab_size} ({args.tokenizer})")
    print(f"Chinchilla Ratio: {config.chinchilla_ratio} tokens per parameter")
    print("=" * 70)
    
    models_to_train = ["hgdm", "transformer"] if args.model == "both" else [args.model]
    
    for m in models_to_train:
        train_single_model(
            model_type=m,
            config=config,
            dataset_name=args.dataset,
            tokenizer_name=args.tokenizer,
            batch_size=args.batch_size,
            accum_steps=args.accum_steps,
            seq_len=args.seq_len,
            lr=args.lr,
            max_tokens=args.max_tokens,
            save_dir=args.save_dir,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
