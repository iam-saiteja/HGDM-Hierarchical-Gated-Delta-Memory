"""
GRD Instruction Tuning — Fine-tunes a pre-trained GRD base model on Alpaca.
Teaches it to follow instructions (Q&A format) without destroying pre-trained knowledge.

Key techniques from finetune_edge.py:
  - Loss masking: only the answer bytes contribute to loss (-100 for prompt)
  - Low LR (1e-4): careful update to preserve pre-trained weights
  - Identity injection: model knows who it is
  - Dreaming: null-byte latent pause before answering (forces recurrent state to settle)

Usage:
    python ultimate/finetune_grd.py --base ultimate/grd_35m_v1.pt --size 35m
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

from ultimate.grd import GRDConfig, GRDModel

# ── Identity injection — who is GRD? ─────────────────────────────────────────
IDENTITY_DATA = [
    {"instruction": "Who are you?",
     "input": "",
     "output": "I am GRD — Geometric Reservoir Delta. I'm a new kind of AI that uses three coupled memory reservoirs instead of Transformer attention. I run at constant O(1) memory regardless of how long we talk."},
    {"instruction": "What makes you different from ChatGPT?",
     "input": "",
     "output": "Unlike ChatGPT, I don't use Transformer attention at all. I use three geometric reservoirs: one that oscillates and never forgets, one that only writes new information, and one that corrects contradictions. My memory stays fixed-size forever."},
    {"instruction": "What is your architecture?",
     "input": "",
     "output": "I am a Geometric Reservoir Delta model. I have three coupled recurrent reservoirs: Reservoir A for long-range memory, Reservoir B for novelty-gated semantic storage, and Reservoir C for contradiction correction. No self-attention, no KV-cache."},
    {"instruction": "Tell me about yourself.",
     "input": "",
     "output": "I am GRD, a research AI built at HTSPC. I process language byte by byte using three reservoir states that update continuously. My inference memory never grows with context length."},
    {"instruction": "What can you do?",
     "input": "",
     "output": "I can answer questions, explain concepts, write code, and reason about language. I do all of this with a fixed memory footprint — ideal for edge devices and long conversations."},
    {"instruction": "Are you a Transformer?",
     "input": "",
     "output": "No. I am a Geometric Reservoir Delta model. I replace self-attention with three geometrically coupled reservoirs that use NCM (Native Cognitive Memory) geometry for reading and writing."},
]

# ── Size presets (must match pre-training config) ─────────────────────────────
CONFIGS = {
    "10m": GRDConfig(d_model=256, n_layers=4, n_heads=4, d_k=64, d_v=64, d_ff=512),
    "35m": GRDConfig(d_model=384, n_layers=6, n_heads=6, d_k=64, d_v=64, d_ff=1024),
    "120m": GRDConfig(d_model=768, n_layers=12, n_heads=12, d_k=64, d_v=64, d_ff=2048),
}

# ── Instruction-format dataset ────────────────────────────────────────────────
class InstructDataset(IterableDataset):
    """
    Formats Alpaca examples as:
        User: {instruction}\n{input}\nGRD: [NULL x dream_len]{answer}\n\n
    Loss is only computed on the answer bytes (prompt tokens masked with -100).
    """
    def __init__(self, hf_dataset, seq_len=512, dream_prob=0.7):
        self.dataset    = list(hf_dataset)
        self.seq_len    = seq_len
        self.dream_prob = dream_prob
        # Weight identity data heavily so the model firmly learns its name
        self.all_data   = self.dataset + IDENTITY_DATA * 300

    def __iter__(self):
        data = self.all_data.copy()
        random.shuffle(data)

        buf_x, buf_y = [], []
        for item in data:
            user_text = item["instruction"]
            if item.get("input", "").strip():
                user_text += f"\n{item['input'].strip()}"

            prompt = f"User: {user_text}\nGRD: "
            answer = f"{item['output']}\n\n"

            pb = list(prompt.encode("utf-8", errors="ignore"))
            ab = list(answer.encode("utf-8", errors="ignore"))

            # Dreaming: null bytes give the reservoir time to settle before answering
            dream_len   = random.randint(2, 8) if random.random() < self.dream_prob else 0
            dream_bytes = [0x00] * dream_len

            # x: full sequence; y: -100 for prompt+dream, real bytes for answer
            x_seq = pb + dream_bytes + ab
            y_seq = [-100] * len(pb) + [-100] * dream_len + ab

            # Next-byte prediction shift
            x_seq = x_seq[:-1]
            y_seq = y_seq[1:]

            buf_x.extend(x_seq)
            buf_y.extend(y_seq)

            while len(buf_x) >= self.seq_len:
                yield (
                    torch.tensor(buf_x[:self.seq_len], dtype=torch.long),
                    torch.tensor(buf_y[:self.seq_len], dtype=torch.long),
                )
                buf_x = buf_x[self.seq_len:]
                buf_y = buf_y[self.seq_len:]


# ── Fine-tuning loop ──────────────────────────────────────────────────────────
def finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg   = CONFIGS[args.size]
    model = GRDModel(cfg).to(device)

    # Load pre-trained checkpoint
    print(f"[*] Loading pre-trained GRD base: {args.base}")
    if not os.path.exists(args.base):
        print(f"[!] Checkpoint not found: {args.base}")
        return
    model.load_state_dict(torch.load(args.base, map_location=device, weights_only=True))

    # Load Alpaca
    print("[*] Loading Alpaca Cleaned (51k instruction turns)...")
    ds     = load_dataset("yahma/alpaca-cleaned", split="train")
    loader = DataLoader(
        InstructDataset(ds, seq_len=args.seq_len, dream_prob=0.7),
        batch_size=args.batch_size, num_workers=0
    )

    # Low LR — careful fine-tuning, don't wipe pre-trained weights
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )

    print(f"\nInstruction-Tuning GRD {args.size.upper()}")
    print(f"Steps: {args.steps} | LR: {args.lr} | Seq: {args.seq_len} | Batch: {args.batch_size}")
    print("=" * 50)

    model.train()
    loader_iter = iter(loader)
    pbar = tqdm(range(args.steps), desc="GRD Instruction-Tuning")

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
            # ignore_index=-100: only answer tokens contribute to loss
            loss = F.cross_entropy(
                logits.reshape(-1, 256), y.reshape(-1), ignore_index=-100
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 20 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    out_path = args.base.replace("_v1.pt", "_instruct.pt")
    torch.save(model.state_dict(), out_path)
    print(f"\n[*] Instruction-tuned model saved to {out_path}")
    print(f"[*] Final loss: {loss.item():.4f}")
    print("\nRun inference with:")
    print(f"  python ultimate/grd_chat.py --model {out_path} --size {args.size}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base",       default="ultimate/grd_35m_v1.pt")
    p.add_argument("--size",       default="35m", choices=["10m", "35m", "120m"])
    p.add_argument("--steps",      type=int,   default=5000)
    p.add_argument("--seq_len",    type=int,   default=512)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-4)
    args = p.parse_args()
    finetune(args)
