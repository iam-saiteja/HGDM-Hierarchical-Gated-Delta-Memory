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
from tokenized.dataset import StatefulStreamDataset


def get_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_states_for_mask(states: List[Tuple], reset_mask: torch.Tensor):
    """
    Zeroes out state entries in S, n, last_k for batch indices where reset_mask is True.
    """
    if states is None or not reset_mask.any():
        return states
    
    new_states = []
    for layer_st in states:
        if len(layer_st) == 3:
            S, n, last_k = layer_st
            S = S.clone()
            S[reset_mask] = 0.0
            if n is not None:
                n = n.clone()
                n[reset_mask] = 0.0
            if last_k is not None:
                last_k = last_k.clone()
                last_k[reset_mask] = 0.0
            new_states.append((S, n, last_k))
        elif len(layer_st) == 2:
            S, n = layer_st
            S = S.clone()
            S[reset_mask] = 0.0
            if n is not None:
                n = n.clone()
                n[reset_mask] = 0.0
            new_states.append((S, n))
    return new_states


def detach_states(states: List[Tuple]):
    """
    Detaches recurrent states for Truncated Backpropagation Through Time (TBPTT).
    """
    if states is None:
        return None
    return [
        tuple(t.detach() if t is not None else None for t in layer_st)
        for layer_st in states
    ]


def evaluate_generation(model: TokenizedHGDM, tokenizer, device: torch.device):
    """
    Runs quick qualitative test sequence generation to verify anti-repetition.
    """
    test_prompts = [
        "User: What is artificial intelligence?\nAssistant:",
        "Count numbers: 1, 2, 3, 4,",
        "The primary advantage of recurrent memory is",
    ]
    print("\n" + "=" * 60)
    print("  QUALITATIVE GENERATION CHECK")
    print("=" * 60)
    model.eval()
    for prompt in test_prompts:
        p_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        out_ids = model.generate(p_ids, max_new_tokens=40, temperature=0.7, top_k=40, repetition_penalty=1.15)
        text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
        print(f"PROMPT: {prompt.strip()}")
        print(f"OUTPUT: {text.strip()}")
        print("-" * 60)
    print("=" * 60 + "\n")
    model.train()


def train_recurrent_native(
    config: TokenizedConfig,
    dataset_name: str = "HuggingFaceTB/smoltalk",
    tokenizer_name: str = "gpt2",
    batch_size: int = 4,
    accum_steps: int = 4,
    seq_len: int = 1024,
    lr: float = 5e-4,
    max_tokens: int = 100_000_000,
    save_dir: str = "checkpoints_tokenized/recurrent_native",
    device_str: str = "cuda",
    save_every: int = 1000,
    eval_every: int = 500,
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Instantiate Model with Recurrent-Native Settings
    config.use_hbs = True
    config.hbs_n_bands = 8
    config.lateral_inhibition_gamma = 0.35
    config.use_mtp = True
    config.stateful_carryover = True

    print("=" * 75)
    print("RECURRENT-NATIVE TRAINING PARADIGM (HGDM-HBS 8-BAND)")
    print("=" * 75)
    print("Core Mechanisms:")
    print("  1. 8 Dyadic Bands (K=8, strides 1 to 128, horizon up to 500,000 tokens)")
    print("  2. GABAergic Lateral Self-Inhibition (gamma=0.35)")
    print("  3. Stateful Document Streaming (Continuous TBPTT across chunks)")
    print("  4. Multi-Token Prediction (MTP auxiliary heads t+2, t+4)")
    print("=" * 75)
    
    model = TokenizedHGDM(config).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokens_per_step = batch_size * accum_steps * seq_len
    total_steps = math.ceil(max_tokens / tokens_per_step)

    print(f"Trainable Parameters: {param_count:,}")
    print(f"Target Token Budget: {max_tokens:,} tokens (~{max_tokens / 1e6:.1f}M tokens)")
    print(f"Batch Configuration: {batch_size} streams x {seq_len} seq_len x {accum_steps} accum = {tokens_per_step:,} tok/step")
    print(f"Total Steps: {total_steps:,} steps")
    print(f"Save Directory: {save_dir}")
    print("-" * 75)

    # 2. Stateful Dataset
    dataset = StatefulStreamDataset(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        batch_size=batch_size,
        seq_len=seq_len,
        max_tokens=max_tokens,
    )
    tokenizer = dataset.tokenizer

    # 3. Optimizer & Cosine Schedule
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=lr * 0.1
    )
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    # Persistent Recurrent State across batch streams
    recurrent_states = None
    
    model.train()
    tokens_processed = 0
    step = 0
    t0 = time.time()
    history = []

    print("[Training Started] Streaming continuous documents...")

    while tokens_processed < max_tokens and step < total_steps:
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_loss_1 = 0.0
        accum_loss_2 = 0.0
        accum_loss_4 = 0.0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for _ in range(accum_steps):
            try:
                batch = next(dataset)
            except StopIteration:
                break

            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            reset_mask = batch["reset_mask"].to(device)

            # Apply reset_mask to zero out states of finished documents
            recurrent_states = reset_states_for_mask(recurrent_states, reset_mask)

            with autocast(device_type=device.type, dtype=torch.bfloat16):
                # Forward pass with persistent state carryover and MTP
                out, next_states = model(x, states=recurrent_states, return_mtp=True)

                if config.use_mtp and isinstance(out, tuple):
                    logits_1, logits_2, logits_4 = out
                    # Primary next-token loss (t+1)
                    l1 = F.cross_entropy(logits_1.reshape(-1, config.vocab_size), y.reshape(-1))
                    # Auxiliary MTP losses: t+2 and t+4
                    l2 = F.cross_entropy(logits_2[:, :-1].reshape(-1, config.vocab_size), y[:, 1:].reshape(-1))
                    l4 = F.cross_entropy(logits_4[:, :-3].reshape(-1, config.vocab_size), y[:, 3:].reshape(-1))
                    loss = l1 + 0.3 * l2 + 0.1 * l4
                else:
                    logits_1 = out
                    l1 = F.cross_entropy(logits_1.reshape(-1, config.vocab_size), y.reshape(-1))
                    l2 = torch.tensor(0.0)
                    l4 = torch.tensor(0.0)
                    loss = l1

                loss_scaled = loss / accum_steps

            if scaler:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            accum_loss += loss.item()
            accum_loss_1 += l1.item()
            accum_loss_2 += l2.item() if isinstance(l2, torch.Tensor) else l2
            accum_loss_4 += l4.item() if isinstance(l4, torch.Tensor) else l4
            tokens_processed += x.numel()

            # Detach state for continuous TBPTT to the next chunk
            recurrent_states = detach_states(next_states)

        accum_loss /= accum_steps
        accum_loss_1 /= accum_steps
        accum_loss_2 /= accum_steps
        accum_loss_4 /= accum_steps

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

        # Periodic Logging
        if step % 20 == 0 or step == 1 or tokens_processed >= max_tokens:
            elapsed = time.time() - t0
            tok_per_sec = tokens_processed / max(1e-4, elapsed)
            ppl = math.exp(min(20.0, accum_loss_1))
            vram = get_vram_mb()
            cur_lr = opt.param_groups[0]['lr']

            print(
                f"Step {step:4d}/{total_steps} | "
                f"Tokens: {tokens_processed:>9,}/{max_tokens:,} ({tokens_processed/max_tokens*100:4.1f}%) | "
                f"Loss: {accum_loss:.4f} (L1: {accum_loss_1:.4f}, L2: {accum_loss_2:.4f}, L4: {accum_loss_4:.4f}) | "
                f"PPL: {ppl:6.2f} | Throughput: {tok_per_sec:7.0f} tok/s | VRAM: {vram:5.0f}MB | LR: {cur_lr:.2e}"
            )

            history.append({
                "step": step,
                "tokens": tokens_processed,
                "loss_total": accum_loss,
                "loss_l1": accum_loss_1,
                "loss_l2": accum_loss_2,
                "loss_l4": accum_loss_4,
                "ppl": ppl,
                "tok_per_sec": tok_per_sec,
                "vram_mb": vram,
                "time_s": elapsed,
            })

        # Periodic Evaluation
        if step % eval_every == 0:
            evaluate_generation(model, tokenizer, device)

        # Periodic Checkpoint
        if step % save_every == 0 or tokens_processed >= max_tokens:
            ckpt_path = os.path.join(save_dir, "ckpt_hgdm.pt")
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "model_type": "hgdm_recurrent_native",
                "tokens_processed": tokens_processed,
                "step": step,
                "history": history,
            }, ckpt_path)
            print(f"[Saved Periodic Checkpoint] {ckpt_path} at step {step}")

    # Final Save
    final_ckpt = os.path.join(save_dir, "ckpt_hgdm.pt")
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "model_type": "hgdm_recurrent_native",
        "tokens_processed": tokens_processed,
        "step": step,
        "history": history,
    }, final_ckpt)
    print(f"\n[Training Complete] Final checkpoint saved: {final_ckpt}")

    # Final Evaluation
    evaluate_generation(model, tokenizer, device)
    return final_ckpt, history


def main():
    parser = argparse.ArgumentParser(description="Recurrent-Native HGDM Trainer (8-Band HBS, TBPTT, MTP)")
    parser.add_argument("--tier", type=str, choices=["pilot", "standard", "paper"], default="pilot")
    parser.add_argument("--dataset", type=str, default="HuggingFaceTB/smoltalk")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_tokens", type=int, default=100_000_000, help="Target training tokens (default 100M)")
    parser.add_argument("--save_dir", type=str, default="checkpoints_tokenized/recurrent_native")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=500)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = tok.vocab_size

    config = TokenizedConfig.get_tier(args.tier, vocab_size=vocab_size)
    config.use_hbs = True
    config.hbs_n_bands = 8
    config.lateral_inhibition_gamma = 0.35
    config.use_mtp = True
    config.stateful_carryover = True

    train_recurrent_native(
        config=config,
        dataset_name=args.dataset,
        tokenizer_name=args.tokenizer,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        seq_len=args.seq_len,
        lr=args.lr,
        max_tokens=args.max_tokens,
        save_dir=args.save_dir,
        save_every=args.save_every,
        eval_every=args.eval_every,
    )


if __name__ == "__main__":
    main()
