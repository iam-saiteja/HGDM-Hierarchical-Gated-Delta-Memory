import os
import sys
import time
import math
import argparse
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from typing import Optional, Iterator, Dict, List
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenized.config import TokenizedConfig
from tokenized.model_hgdm import TokenizedHGDM
from tokenized.model_transformer import MatchedTransformer

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


class SFTStreamingDataset:
    """
    Supervised Fine-Tuning Dataset with Prompt Loss Masking.
    Masks user tokens with loss_mask = 0.0,
    Computes loss ONLY on assistant response tokens with loss_mask = 1.0.
    """
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        seq_len: int = 512,
        dataset_name: str = "HuggingFaceTB/smoltalk",
        subset: str = "all",
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.eos_id = tokenizer.eos_token_id or 0

    def _stream_conversations(self) -> Iterator[List[Dict[str, str]]]:
        if load_dataset is not None:
            try:
                ds = load_dataset(self.dataset_name, self.subset, split=self.split, streaming=True)
                for row in ds:
                    if "messages" in row and isinstance(row["messages"], list):
                        yield row["messages"]
            except Exception as e:
                print(f"[SFT Data] Streaming failed ({e}). Falling back to clean conversational dataset.")
                
        # Clean fallback dialogues if offline or streaming drops
        fallback_data = [
            [
                {"role": "user", "content": "What is artificial intelligence and how does it work?"},
                {"role": "assistant", "content": "Artificial intelligence refers to computer systems designed to perform cognitive tasks such as reasoning, learning, and pattern recognition. It works by training algorithms on massive datasets to adjust internal mathematical parameters to minimize prediction errors."}
            ],
            [
                {"role": "user", "content": "Write a python function to add two numbers."},
                {"role": "assistant", "content": "Here is a clean Python function to add two numbers:\n\n```python\ndef add_two_numbers(a, b):\n    return a + b\n\n# Example usage:\nprint(add_two_numbers(10, 20))  # Output: 30\n```"}
            ],
            [
                {"role": "user", "content": "Explain the difference between supervised learning and unsupervised learning in machine learning."},
                {"role": "assistant", "content": "In supervised learning, the model is trained on labeled data where each input is paired with the correct output (e.g. classification, regression). In unsupervised learning, the model discovers hidden patterns, clusters, or latent structures in unlabeled data without predefined ground truth."}
            ],
            [
                {"role": "user", "content": "Here is a list of secret agent code names: Alice is Falcon, Bob is Shadow, Charlie is Viper, David is Ghost. Question: What is the secret code name of Charlie? Answer:"},
                {"role": "assistant", "content": "The secret code name of Charlie is Viper."}
            ],
            [
                {"role": "user", "content": "The library was silent. Outside the rain fell against the high stained-glass windows. Important note: The hidden gate code is 7492. Many researchers sat reading old encyclopedias and discussing philosophy until dawn. Question: What is the hidden gate code? Answer:"},
                {"role": "assistant", "content": "The hidden gate code is 7492."}
            ],
            [
                {"role": "user", "content": "Count from 1 to 20 separated by commas: 1, 2, 3, 4, 5,"},
                {"role": "assistant", "content": " 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20."}
            ],
            [
                {"role": "user", "content": "Reverse the order of these words: orange, banana, apple, grape, cherry. Output:"},
                {"role": "assistant", "content": "cherry, grape, apple, banana, orange"}
            ],
            [
                {"role": "user", "content": "Once upon a time in a faraway kingdom,"},
                {"role": "assistant", "content": "there lived a humble clockmaker whose mechanical birds could sing melodies that healed troubled hearts across the land."}
            ]
        ]
        while True:
            for item in fallback_data:
                yield item

    def get_batches(self, batch_size: int, device: torch.device) -> Iterator[Dict[str, torch.Tensor]]:
        input_buffer = []
        target_buffer = []
        mask_buffer = []

        for messages in self._stream_conversations():
            # Build prompt and assistant completion
            prompt_str = ""
            completion_str = ""
            
            for msg in messages:
                role = msg.get("role", "").lower()
                content = msg.get("content", "").strip()
                if not content:
                    continue
                if role in ["user", "human"]:
                    prompt_str += f"User: {content}\nAssistant:"
                elif role in ["assistant", "bot"]:
                    completion_str += f" {content}" + self.tokenizer.eos_token
                    
            if not prompt_str or not completion_str:
                continue

            prompt_ids = self.tokenizer.encode(prompt_str)
            comp_ids = self.tokenizer.encode(completion_str)
            
            combined_ids = prompt_ids + comp_ids
            # Mask: 0 for prompt, 1 for completion
            combined_mask = [0.0] * len(prompt_ids) + [1.0] * len(comp_ids)

            if len(combined_ids) < 4:
                continue

            # Truncate if exceeds seq_len + 1
            if len(combined_ids) > self.seq_len + 1:
                combined_ids = combined_ids[: self.seq_len + 1]
                combined_mask = combined_mask[: self.seq_len + 1]

            # Autoregressive shift: x = combined_ids[:-1], y = combined_ids[1:]
            x_seq = combined_ids[:-1]
            y_seq = combined_ids[1:]
            m_seq = combined_mask[1:]  # Mask corresponds to target y_seq

            # Pad to seq_len
            pad_len = self.seq_len - len(x_seq)
            if pad_len > 0:
                x_seq = x_seq + [self.tokenizer.pad_token_id or 0] * pad_len
                y_seq = y_seq + [self.tokenizer.pad_token_id or 0] * pad_len
                m_seq = m_seq + [0.0] * pad_len

            input_buffer.append(x_seq)
            target_buffer.append(y_seq)
            mask_buffer.append(m_seq)

            if len(input_buffer) == batch_size:
                yield {
                    "input_ids": torch.tensor(input_buffer, dtype=torch.long, device=device),
                    "targets": torch.tensor(target_buffer, dtype=torch.long, device=device),
                    "loss_mask": torch.tensor(mask_buffer, dtype=torch.float32, device=device),
                }
                input_buffer = []
                target_buffer = []
                mask_buffer = []


def train_sft(
    model_type: str,
    base_ckpt: str,
    output_ckpt: str,
    steps: int = 500,
    batch_size: int = 8,
    accum_steps: int = 2,
    seq_len: int = 512,
    lr: float = 1e-4,
    device_str: str = "cuda"
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 75)
    print(f"      SUPERVISED FINE-TUNING (SFT) WITH PROMPT MASKING: {model_type.upper()}")
    print("=" * 75)
    print(f"Loading Base Checkpoint: {base_ckpt}")
    
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(f"Checkpoint '{base_ckpt}' not found!")

    data = torch.load(base_ckpt, map_location=device, weights_only=False)
    config = data["config"]

    if model_type.lower() == "hgdm":
        model = TokenizedHGDM(config).to(device)
    else:
        model = MatchedTransformer(config).to(device)

    model.load_state_dict(data["model_state"])
    print(f"Loaded {model_type.upper()} successfully ({sum(p.numel() for p in model.parameters()):,} parameters).")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_ds = SFTStreamingDataset(tokenizer=tokenizer, seq_len=seq_len)
    batch_gen = sft_ds.get_batches(batch_size=batch_size, device=device)

    # Optimizer with Warmup & Cosine decay
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    warmup_steps = 30
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    model.train()
    t0 = time.time()
    total_tokens_trained = 0

    print(f"Training for {steps} steps | Effective Batch: {batch_size * accum_steps * seq_len:,} tok/step")
    print("-" * 75)

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(accum_steps):
            batch = next(batch_gen)
            input_ids = batch["input_ids"]
            targets = batch["targets"]
            loss_mask = batch["loss_mask"]

            with autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, _ = model(input_ids)
                B, T, V = logits.shape
                
                ce = F.cross_entropy(logits.view(-1, V), targets.view(-1), reduction='none')
                # Prompt loss masking: only assistant response tokens are penalized!
                mask_sum = loss_mask.view(-1).sum().clamp(min=1.0)
                loss = (ce * loss_mask.view(-1)).sum() / mask_sum
                loss = loss / accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()
            total_tokens_trained += int(mask_sum.item())

        if scaler is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        scheduler.step()

        if step % 25 == 0 or step == steps:
            elapsed = max(1e-4, time.time() - t0)
            tok_per_sec = total_tokens_trained / elapsed
            ppl = math.exp(min(20.0, accum_loss))
            cur_lr = scheduler.get_last_lr()[0]
            vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
            print(
                f"Step {step:4d}/{steps} | SFT Loss: {accum_loss:6.4f} | "
                f"PPL: {ppl:7.2f} | Supervised Tokens: {total_tokens_trained:9,d} | "
                f"Speed: {tok_per_sec:7.0f} tok/s | VRAM: {vram:5.0f}MB | LR: {cur_lr:.2e}"
            )

    os.makedirs(os.path.dirname(output_ckpt), exist_ok=True)
    torch.save(
        {
            "model_type": model_type,
            "model_state": model.state_dict(),
            "config": config,
            "sft_steps": steps,
            "final_sft_loss": accum_loss,
        },
        output_ckpt
    )
    print(f"\n[Saved] SFT Checkpoint saved to '{output_ckpt}' ({os.path.getsize(output_ckpt)/(1024*1024):.1f} MB)")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning with Prompt Loss Masking")
    parser.add_argument("--model_type", type=str, required=True, choices=["hgdm", "transformer"])
    parser.add_argument("--base_ckpt", type=str, default=None)
    parser.add_argument("--output_ckpt", type=str, default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    base_ckpt = args.base_ckpt or f"checkpoints_tokenized/ckpt_{args.model_type}.pt"
    output_ckpt = args.output_ckpt or f"checkpoints_tokenized/ckpt_{args.model_type}_sft.pt"

    train_sft(
        model_type=args.model_type,
        base_ckpt=base_ckpt,
        output_ckpt=output_ckpt,
        steps=args.steps,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        seq_len=args.seq_len,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
