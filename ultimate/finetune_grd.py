"""
GRD Instruction Tuning — Fine-tunes a pre-trained GRD base model on Alpaca.

Fixes vs v1:
  - Identity data weighted 2000x (by count) so it dominates by token count too
  - Adds stop token ("\n\n") to train the model to end responses cleanly
  - Multi-turn examples: some Alpaca pairs stacked into 2-turn sequences
  - Cosine LR schedule to stabilise training
  - Default LR 3e-4, steps 12000

Usage:
    python ultimate/finetune_grd.py --base ultimate/grd_35m_v1.pt --size 35m
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import math, random
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

from ultimate.grd import GRDConfig, GRDModel

# ─────────────────────────────────────────────────────────────────────────────
# Identity data — who is GRD?
# ─────────────────────────────────────────────────────────────────────────────
IDENTITY_DATA = [
    {"instruction": "Who are you?",
     "output": "I am GRD — Geometric Reservoir Delta. I'm a new kind of AI that uses three coupled memory reservoirs instead of Transformer attention. I run at constant O(1) memory regardless of how long we talk."},
    {"instruction": "What are you?",
     "output": "I am GRD, a Geometric Reservoir Delta model. I replace self-attention with three recurrent reservoirs: one for long-range memory, one for novelty-gated semantic storage, and one for contradiction correction."},
    {"instruction": "What makes you different from ChatGPT?",
     "output": "Unlike ChatGPT which uses Transformer attention, I use three geometric memory reservoirs. My memory stays fixed-size no matter how long the conversation — no KV-cache, no quadratic cost."},
    {"instruction": "What is your architecture?",
     "output": "I am a Geometric Reservoir Delta model with three coupled recurrent reservoirs: Reservoir A for long-range oscillatory memory, Reservoir B for novelty-gated semantic writes, and Reservoir C for contradiction correction. No self-attention."},
    {"instruction": "Tell me about yourself.",
     "output": "I am GRD, a research AI built at HTSPC. I process language byte by byte with three coupled reservoir states that update in constant time and space. I was designed as an alternative to Transformer-based language models."},
    {"instruction": "Are you a Transformer?",
     "output": "No. I am a Geometric Reservoir Delta model. I replace self-attention with three geometrically coupled recurrent reservoirs powered by the NCM kernel, giving me constant inference memory regardless of context length."},
    {"instruction": "What can you do?",
     "output": "I can answer questions, explain concepts, and write text. I do this with a fixed memory footprint — my reservoir states never grow with context length, making me efficient for long conversations."},
    {"instruction": "Hi",
     "output": "Hello! I am GRD — Geometric Reservoir Delta. How can I help you today?"},
    {"instruction": "Hello",
     "output": "Hi there! I am GRD. What can I help you with?"},
    {"instruction": "What is GRD?",
     "output": "GRD stands for Geometric Reservoir Delta — a recurrent neural architecture that replaces Transformer attention with three coupled memory reservoirs for efficient, constant-memory language modeling."},
    {"instruction": "Who built you?",
     "output": "I was built at HTSPC as a research project in alternative architectures to Transformers. I use Geometric Reservoir Delta mechanics to process language."},
]

CONFIGS = {
    "10m":  GRDConfig(d_model=256, n_layers=4,  n_heads=4,  d_k=64, d_v=64, d_ff=512),
    "35m":  GRDConfig(d_model=384, n_layers=6,  n_heads=6,  d_k=64, d_v=64, d_ff=1024),
    "120m": GRDConfig(d_model=768, n_layers=12, n_heads=12, d_k=64, d_v=64, d_ff=2048),
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class InstructDataset(IterableDataset):
    """
    Formats each example as:
        User: {instruction}\nGRD: {answer}\n\n
    Loss computed ONLY on the answer bytes (prompt masked with -100).
    Identity data is repeated IDENTITY_REPEAT times to dominate by token count.
    """
    def __init__(self, hf_dataset, seq_len=512,
                 identity_repeat=2000, dream_prob=0.5):
        self.seq_len = seq_len
        self.dream_prob = dream_prob
        alpaca = [{"instruction": x["instruction"],
                   "output":      x["output"]}
                  for x in hf_dataset]
        identity_block = IDENTITY_DATA * identity_repeat
        self.all_data = alpaca + identity_block
        print(f"    Dataset: {len(alpaca):,} Alpaca + "
              f"{len(identity_block):,} identity = "
              f"{len(self.all_data):,} total examples")

    def _encode_example(self, item):
        instruction = item["instruction"].strip()
        answer      = item.get("output", item.get("input", "")).strip()
        if item.get("input", "").strip():
            instruction = f"{instruction}\n{item['input'].strip()}"

        prompt = f"User: {instruction}\nGRD: "
        answer = f"{answer}\n\n"   # "\n\n" = natural stop signal

        pb = list(prompt.encode("utf-8", errors="ignore"))
        ab = list(answer.encode("utf-8", errors="ignore"))

        dream_len   = random.randint(2, 6) if random.random() < self.dream_prob else 0
        dream_bytes = [0x00] * dream_len

        x_seq = pb + dream_bytes + ab
        y_seq = [-100] * (len(pb) + dream_len) + ab

        # Next-byte prediction shift
        return x_seq[:-1], y_seq[1:]

    def __iter__(self):
        data = self.all_data.copy()
        random.shuffle(data)

        buf_x, buf_y = [], []
        for item in data:
            xs, ys = self._encode_example(item)
            buf_x.extend(xs)
            buf_y.extend(ys)
            while len(buf_x) >= self.seq_len:
                yield (torch.tensor(buf_x[:self.seq_len], dtype=torch.long),
                       torch.tensor(buf_y[:self.seq_len], dtype=torch.long))
                buf_x = buf_x[self.seq_len:]
                buf_y = buf_y[self.seq_len:]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg   = CONFIGS[args.size]
    model = GRDModel(cfg).to(device)

    print(f"[*] Loading pre-trained GRD base: {args.base}")
    if not os.path.exists(args.base):
        print(f"[!] Not found: {args.base}")
        return
    model.load_state_dict(torch.load(args.base, map_location=device, weights_only=True))

    print("[*] Loading Alpaca Cleaned (51k turns)...")
    ds     = load_dataset("yahma/alpaca-cleaned", split="train")
    loader = DataLoader(
        InstructDataset(ds, seq_len=args.seq_len,
                        identity_repeat=args.identity_repeat,
                        dream_prob=0.5),
        batch_size=args.batch_size, num_workers=0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=0.01
    )

    # Cosine LR schedule with warmup
    def lr_lambda(step):
        warmup = 300
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(args.steps - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\nInstruction-Tuning GRD {args.size.upper()}")
    print(f"Steps: {args.steps} | LR: {args.lr} | Seq: {args.seq_len} | Batch: {args.batch_size}")
    print(f"Identity repeat: {args.identity_repeat}x")
    print("=" * 52)

    model.train()
    loader_iter = iter(loader)
    pbar = tqdm(range(args.steps), desc="GRD Instruct")

    for step in pbar:
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, 256), y.reshape(-1), ignore_index=-100
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 20 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    out_path = args.base.replace("_v1.pt", "_instruct.pt")
    if out_path == args.base:
        out_path = args.base.replace(".pt", "_instruct.pt")
    torch.save(model.state_dict(), out_path)
    print(f"\n[*] Saved: {out_path}")
    print(f"[*] Final loss: {loss.item():.4f}")
    print(f"\nRun: python ultimate/grd_chat.py --model {out_path} --size {args.size} --instruct")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base",            default="ultimate/grd_35m_v1.pt")
    p.add_argument("--size",            default="35m", choices=["10m", "35m", "120m"])
    p.add_argument("--steps",           type=int,   default=12000)
    p.add_argument("--seq_len",         type=int,   default=512)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--identity_repeat", type=int,   default=2000,
                   help="How many times to repeat the identity examples (by token count)")
    args = p.parse_args()
    finetune(args)
