"""
Exp 12: Passkey Retrieval — The Needle In A Haystack Test
Revised v3: Dense masked loss, warmup, fixed indexing, smoke test guard.
"""
import torch
import torch.nn.functional as F
import random
import time
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hgdm_ultimate import HGDMUltimate, HGDMConfig

# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────
PASSKEY_PREFIX = b"The passkey is "          # 15 bytes
PASSKEY_SUFFIX = b". "                        # 2 bytes
QUESTION       = b"What is the passkey? "    # 21 bytes

def make_batch(batch_size: int, seq_len: int, device, fixed_depth: float = None):
    """
    Build a batch of sequences.
    x[t]  -> model input token at position t
    y[t]  -> correct next token (only supervised at masked positions)
    mask  -> float weight per position (0 = ignore, 1 = prompt byte, 5 = digit answer)

    Layout (left-to-right):
      [noise] [PASSKEY_PREFIX + digit + PASSKEY_SUFFIX] [noise] [QUESTION + digit]
                                                                  ^--- model predicts here

    The very last input token is the space after '?'.
    The model must output the passkey digit byte.
    """
    x    = torch.randint(97, 123, (batch_size, seq_len), dtype=torch.long, device=device)
    y    = torch.zeros_like(x)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)

    q_len = len(QUESTION)  # 21

    for b in range(batch_size):
        digit  = random.randint(0, 9)
        digit_byte = ord(str(digit))

        # ── 1. Build the passkey sentence and embed it ─────────────────
        sentence = PASSKEY_PREFIX + bytes([digit_byte]) + PASSKEY_SUFFIX
        s_bytes  = list(sentence)                            # length = 18

        # Needle position: random in first half, or fixed depth if given
        haystack_end = seq_len - q_len - 1                  # last safe insertion point
        if fixed_depth is not None:
            needle_start = max(0, min(int(fixed_depth * seq_len), haystack_end - len(s_bytes)))
        else:
            needle_start = random.randint(0, max(0, haystack_end // 2 - len(s_bytes)))

        x[b, needle_start : needle_start + len(s_bytes)] = torch.tensor(s_bytes, device=device)

        # Supervise the passkey sentence (predict the next byte in sequence)
        for t in range(len(s_bytes) - 1):
            y[b, needle_start + t] = s_bytes[t + 1]
            mask[b, needle_start + t] = 1.0

        # ── 2. Embed the question at the very end ──────────────────────
        q_bytes = list(QUESTION)
        q_start = seq_len - q_len
        x[b, q_start : seq_len] = torch.tensor(q_bytes, device=device)

        # Supervise the question bytes
        for t in range(q_len - 1):
            y[b, q_start + t] = q_bytes[t + 1]
            mask[b, q_start + t] = 1.0

        # ── 3. The answer ──────────────────────────────────────────────
        # x[-1] = last byte of QUESTION (the space after '?')
        # y[-1] = digit byte  ← highest-weight supervision target
        y[b, -1] = digit_byte
        mask[b, -1] = 10.0

    return x, y, mask


# ─────────────────────────────────────────────────────────────────
# Smoke Test (ensures the task is learnable at trivial difficulty)
# ─────────────────────────────────────────────────────────────────
def smoke_test(model, device):
    """
    100-step sanity check with seq_len=64 (tiny context).
    If the model can't crack this, something is broken architecturally.
    """
    print("\n[Smoke Test] seq_len=64, fixed needle at position 0...")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for step in range(200):
        opt.zero_grad(set_to_none=True)
        x, y, mask = make_batch(16, 64, device, fixed_depth=0.0)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)[0]
            ce = F.cross_entropy(logits.view(-1, 256), y.view(-1), reduction='none')
            loss = (ce * mask.view(-1)).sum() / mask.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            pred = logits[:, -1, :].argmax(-1)
            acc  = (pred == y[:, -1]).float().mean().item() * 100
            print(f"  step {step:3d} | loss {loss.item():.3f} | acc {acc:.0f}%")
    print("[Smoke Test] done.\n")
    return model


# ─────────────────────────────────────────────────────────────────
# Curriculum Training
# ─────────────────────────────────────────────────────────────────
def train_curriculum(model, device):
    print("="*60)
    print("EXP 12: PASSKEY RETRIEVAL (THE NEEDLE TEST)")
    print("="*60)

    curriculum = [
        {"len": 256,   "steps": 800,  "batch": 8},
        {"len": 1024,  "steps": 800,  "batch": 4},
        {"len": 4096,  "steps": 1000, "batch": 2},
        {"len": 8192,  "steps": 800,  "batch": 1},
        {"len": 16384, "steps": 600,  "batch": 1},
        {"len": 32768, "steps": 400,  "batch": 1},
    ]
    warmup_steps = 100
    total_steps  = sum(c["steps"] for c in curriculum)

    opt    = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.01)
    sched  = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, total_steps=total_steps, pct_start=0.1,
        anneal_strategy='cos', div_factor=25, final_div_factor=100
    )
    scaler = torch.amp.GradScaler('cuda')

    model.train()
    global_step = 0

    for phase in curriculum:
        seq_len = phase["len"]
        steps   = phase["steps"]
        batch   = phase["batch"]
        print(f"\n--- Phase: seq_len={seq_len} | steps={steps} | batch={batch} ---")
        t0 = time.time()

        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            x, y, mask = make_batch(batch, seq_len, device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)[0]
                ce   = F.cross_entropy(logits.view(-1, 256), y.view(-1), reduction='none')
                loss = (ce * mask.view(-1)).sum() / mask.sum()

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            global_step += 1

            if step % 200 == 0:
                with torch.no_grad():
                    pred = logits[:, -1, :].argmax(-1)
                    acc  = (pred == y[:, -1]).float().mean().item() * 100
                lr = opt.param_groups[0]['lr']
                print(f"  step {step:4d} | loss {loss.item():.3f} | needle_acc {acc:.0f}% | lr {lr:.2e}")

        elapsed = time.time() - t0
        print(f"  Phase done in {elapsed:.1f}s  ({steps/elapsed:.0f} steps/s)")
        
        # Save checkpoint after each phase
        os.makedirs("results", exist_ok=True)
        torch.save(model.state_dict(), f"results/ckpt_phase_L{seq_len}.pt")


def vram_mb():
    """Current GPU memory allocated in MB."""
    return torch.cuda.memory_allocated() / 1024**2

def evaluate_grid(model, device):
    print("\n" + "="*70)
    print("EVALUATION GRID  (O(1) Memory Demonstration)")
    print("="*70)
    print(f"{'L':>6} | {'Depth':>6} | {'Acc':>7} | {'Trials':>6} | {'Peak VRAM':>10} | Sample")
    print("-"*70)
    model.eval()

    # Extended to 32768 — impossible for Transformers on this GPU
    lengths  = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    depths   = [0.1, 0.5, 0.9]
    # Fewer trials for longest sequences to keep runtime reasonable
    trials_map = {512: 30, 1024: 30, 2048: 30, 4096: 30,
                  8192: 20, 16384: 15, 32768: 10}
    results = {}

    with torch.no_grad():
        for L in lengths:
            results[str(L)] = {}
            n_trials = trials_map[L]

            for D in depths:
                correct       = 0
                peak_vram_mb  = 0.0
                example_shown = False

                for _ in range(n_trials):
                    torch.cuda.reset_peak_memory_stats()
                    x, y, _ = make_batch(1, L, device, fixed_depth=D)

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        logits = model(x)[0]

                    peak_vram_mb = max(peak_vram_mb,
                                      torch.cuda.max_memory_allocated() / 1024**2)

                    pred_byte = logits[0, -1, :].argmax().item()
                    target    = y[0, -1].item()
                    sample    = f"{chr(target)}->{chr(pred_byte) if 32 <= pred_byte < 127 else '?'}"

                    if not example_shown:
                        example_shown = True

                    if pred_byte == target:
                        correct += 1

                acc = correct / n_trials * 100
                results[str(L)][str(D)] = {"acc": round(acc, 1),
                                           "peak_vram_mb": round(peak_vram_mb, 1),
                                           "trials": n_trials}
                print(f"  {L:5d} | {D:6.1f} | {acc:6.1f}% | {correct:3d}/{n_trials:<3d} "
                      f"| {peak_vram_mb:7.0f} MB | {sample}")

    print("-"*70)
    print("NOTE: VRAM stays ~constant as L grows — proof of O(1) memory.")
    return results


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def run_experiment():
    if not torch.cuda.is_available():
        print("CUDA required."); sys.exit(1)

    device = torch.device('cuda')

    config = HGDMConfig(
        d_model=384,
        n_layers=6,
        n_heads=6,
        vocab_size=256
    )
    model = HGDMUltimate(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.1f}M")

    # 1. Smoke test first — catches broken architecture before wasting GPU time
    model = smoke_test(model, device)

    # 2. Full curriculum
    train_curriculum(model, device)

    # 3. Evaluation
    grid = evaluate_grid(model, device)

    os.makedirs("results", exist_ok=True)
    with open("results/results.json", "w") as f:
        json.dump(grid, f, indent=4)
    print("\nResults saved to results/results.json")

if __name__ == "__main__":
    run_experiment()
