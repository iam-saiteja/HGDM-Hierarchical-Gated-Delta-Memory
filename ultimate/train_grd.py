"""
GRD Training Script — Train a Geometric Reservoir Delta model on OpenWebText.
Comparable config to HGDM Chinchilla baseline for fair BPB comparison.

Usage:
    python ultimate/train_grd.py                  # 35M model
    python ultimate/train_grd.py --size 120m      # 120M model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys, time, math, argparse
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultimate.grd import GRDConfig, GRDModel, count_parameters

# ─────────────────────────────────────────────────────────────────────────────
# Configs matching Chinchilla baselines for fair comparison
# ─────────────────────────────────────────────────────────────────────────────
CONFIGS = {
    "10m": GRDConfig(d_model=256, n_layers=4, n_heads=4, d_k=64, d_v=64, d_ff=512),
    "35m": GRDConfig(d_model=384, n_layers=6, n_heads=6, d_k=64, d_v=64, d_ff=1024),
    "120m": GRDConfig(d_model=768, n_layers=12, n_heads=12, d_k=64, d_v=64, d_ff=2048),
}

# ─────────────────────────────────────────────────────────────────────────────
# Streaming byte dataset
# ─────────────────────────────────────────────────────────────────────────────
class StreamingByteDataset(IterableDataset):
    def __init__(self, seq_len=1024, max_tokens=None):
        from datasets import load_dataset
        self.ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        self.seq_len = seq_len
        self.max_tokens = max_tokens

    def __iter__(self):
        buf = []
        tokens_yielded = 0
        for sample in self.ds:
            buf.extend(sample["text"].encode("utf-8", errors="ignore"))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[:self.seq_len + 1]
                buf   = buf[self.seq_len + 1:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:],  dtype=torch.long)
                yield x, y
                tokens_yielded += self.seq_len
                if self.max_tokens and tokens_yielded >= self.max_tokens:
                    return

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg   = CONFIGS[args.size]
    model = GRDModel(cfg).to(device)
    total = count_parameters(model)
    print(f"\nGRD {args.size.upper()} | {total:,} parameters")

    # Chinchilla-optimal tokens: 20x params
    chinchilla_tokens = 20 * total
    seq_len    = args.seq_len
    batch_size = args.batch_size
    steps      = chinchilla_tokens // (seq_len * batch_size)
    if args.max_steps > 0:
        steps = min(steps, args.max_steps)
    lr = 3e-4

    print(f"Chinchilla budget: {chinchilla_tokens:,} tokens")
    print(f"Steps: {steps:,} | Batch: {batch_size} | Seq: {seq_len} | Chunk: {args.chunk_size}")
    print(f"LR: {lr} | Warmup: 500 steps")
    if args.max_steps > 0:
        print(f"[!] Capped at --max_steps {args.max_steps} (quick benchmark mode)\n")
    else:
        print()

    ds     = StreamingByteDataset(seq_len=seq_len, max_tokens=chinchilla_tokens + seq_len * batch_size)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    # torch.compile: fuses Python-level einsum/norm ops into a single CUDA kernel
    # This is critical for recurrent models where the Python loop is the bottleneck.
    print("[*] Compiling model with torch.compile (first step will be slow)...")
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        print(f"[!] torch.compile failed ({e}), using eager mode.")
        compiled_model = model

    # Cosine LR schedule with warmup
    def lr_lambda(step):
        warmup = 500
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(steps - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Truncated BPTT helper ────────────────────────────────────────────
    def detach_states(states):
        """Detach all state tensors so gradients don't propagate across chunks."""
        if states is None:
            return None
        result = []
        for state in states:
            if state is None:
                result.append(None)
            else:
                result.append(tuple(s.detach() if torch.is_tensor(s) else s for s in state))
        return result

    CHUNK = args.chunk_size   # backprop through this many steps at a time

    model.train()
    loader_iter = iter(loader)
    pbar = tqdm(range(steps), desc=f"GRD {args.size.upper()}")
    log  = []

    for step in pbar:
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)
        T_total = x.shape[1]
        n_chunks = max(1, T_total // CHUNK)

        optimizer.zero_grad(set_to_none=True)
        chunk_states = None
        total_loss = 0.0

        # ── Truncated BPTT: forward-backward in CHUNK-sized windows ──────
        for c in range(n_chunks):
            x_c = x[:, c * CHUNK : (c + 1) * CHUNK]
            y_c = y[:, c * CHUNK : (c + 1) * CHUNK]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, chunk_states = compiled_model(x_c, chunk_states)
                loss = F.cross_entropy(
                    logits.reshape(-1, 256), y_c.reshape(-1)
                ) / n_chunks   # scale so total gradient ≈ full-seq gradient

            loss.backward()
            total_loss += loss.item()

            # Detach states: stop gradients at chunk boundary
            chunk_states = detach_states(chunk_states)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Rescale for logging (total_loss is already sum of per-chunk losses / n_chunks)
        bpb = total_loss / math.log(2)
        if step % 50 == 0:
            pbar.set_postfix({"loss": f"{total_loss:.4f}", "bpb": f"{bpb:.4f}"})
        if step % 500 == 0:
            log.append({"step": step, "loss": round(total_loss, 4), "bpb": round(bpb, 4)})

    # Save
    out_path = os.path.join(os.path.dirname(__file__), f"grd_{args.size}_v1.pt")
    torch.save(model.state_dict(), out_path)
    print(f"\n[*] Saved to {out_path}")
    print(f"[*] Final loss: {loss.item():.4f} | BPB: {bpb:.4f}")

    # Print log
    print("\nTraining log:")
    for entry in log:
        print(f"  Step {entry['step']:5d} | loss {entry['loss']} | bpb {entry['bpb']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="35m", choices=["10m", "35m", "120m"])
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size. Keep low (4-8) for recurrent BPTT memory.")
    parser.add_argument("--seq_len", type=int, default=256,
                        help="Sequence length. Shorter = faster per step. Default 256 for speed.")
    parser.add_argument("--chunk_size", type=int, default=64,
                        help="Truncated BPTT window. Lower = less VRAM. Should be <= seq_len.")
    parser.add_argument("--max_steps", type=int, default=3000,
                        help="Cap steps for quick benchmark. 0 = full Chinchilla budget.")
    args = parser.parse_args()
    train(args)
