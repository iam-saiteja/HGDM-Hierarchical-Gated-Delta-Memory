import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import subprocess
import math
import json
import sys
from hgdm_ultimate import HGDMUltimate, HGDMConfig
from hgdm_omega import OmegaGDM, OmegaConfig
from data_1b import get_1b_dataloader

PREOCCUPIED_MEM = 0  # Full GPU now available — no preoccupied memory to subtract

def get_net_gpu_memory():
    """Queries nvidia-smi and subtracts preoccupied baseline (11,570 MB)."""
    try:
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        total_used = int(output)
        net_used = max(0, total_used - PREOCCUPIED_MEM)
        return net_used
    except Exception:
        return -1

def get_gpu_temp():
    try:
        cmd = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        return f"{output}C"
    except Exception:
        return "N/A"

def verify_datasets():
    from datasets import load_dataset
    print("[Dataset] Running dataset split pre-start verification...")
    try:
        fw = next(iter(load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)))
        print(f"[Dataset] FineWeb-Edu verified! Sample text len: {len(fw.get('text', ''))}")
        wiki = next(iter(load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)))
        print(f"[Dataset] Wikipedia verified! Sample title: {wiki.get('title', 'N/A')}")
        code = next(iter(load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)))
        print(f"[Dataset] CodeParrot-clean verified! Sample content len: {len(code.get('content', ''))}")
        print("[Dataset] All streaming pipelines successfully connected and verified!")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Dataset verification failed: {e}")
        sys.exit(1)

def run_training_sprint(model, model_name, max_steps, grad_accum_steps, batch_size, block_size, device):
    """
    Trains the given model for max_steps steps on a freshly initialized data stream.
    Returns a list of log dicts {step, loss, bpb, vram_mb, step_time}.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=1e-5)

    # Fresh data stream for each model to ensure identical data exposure
    dataloader = get_1b_dataloader(block_size=block_size, batch_size=batch_size)
    data_stream = iter(dataloader)

    model.train()
    logs = []
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*60}")
    print(f"{'Step':<5} | {'Loss':<10} | {'BPB':<7} | {'Net VRAM':<10} | {'StepTime':<9} | {'Elapsed'}")
    print(f"{'-'*65}")
    sys.stdout.flush()

    for step in range(max_steps):
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        t_step = time.time()

        for _ in range(grad_accum_steps):
            batch = next(data_stream).to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)) / grad_accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[ERROR] NaN/Inf loss at step {step}! Stopping.")
                return logs

            loss.backward()
            accum_loss += loss.item() * grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        step_time = time.time() - t_step
        bpb = accum_loss / math.log(2)

        # Query nvidia-smi only every 5 steps to avoid overhead
        vram_mb = get_net_gpu_memory() if step % 5 == 0 else (logs[-1]['vram_mb'] if logs else -1)

        logs.append({
            "step": step,
            "loss": accum_loss,
            "bpb": bpb,
            "vram_mb": vram_mb,
            "step_time": step_time
        })

        if step % 25 == 0 or step == max_steps - 1:
            elapsed = (time.time() - t_start) / 60
            temp = get_gpu_temp()
            print(f"{step:04d} | {accum_loss:<10.4f} | {bpb:<7.4f} | {vram_mb:<7}MB   | {step_time:.2f}s     | {elapsed:.1f}min  ({temp})")
            sys.stdout.flush()

    return logs


PROMPTS = [
    "The theory of relativity states that",
    "In machine learning, a transformer model",
    "def fibonacci(n):\n    ",
    "The capital of France is Paris. The capital of Germany is",
]

def run_generation_test(model, model_name, device, max_new_bytes=150, temp=0.8):
    """Runs generation on a fixed set of prompts and prints decoded output."""
    model.eval()
    print(f"\n{'='*60}")
    print(f"GENERATION TEST: {model_name}")
    print(f"{'='*60}")

    for i, prompt_text in enumerate(PROMPTS):
        prompt_bytes = list(prompt_text.encode('utf-8', errors='ignore'))
        prompt_tensor = torch.tensor([prompt_bytes], dtype=torch.long, device=device)

        try:
            with torch.no_grad():
                generated = model.generate(prompt_tensor, max_new_bytes=max_new_bytes, temp=temp)
            # Decode only the newly generated bytes (skip prompt)
            new_bytes = generated[0, len(prompt_bytes):].tolist()
            decoded = bytes(new_bytes).decode('utf-8', errors='replace')
        except Exception as e:
            decoded = f"[ERROR during generation: {e}]"

        print(f"\n--- Prompt {i+1} ---")
        print(f"PROMPT : {prompt_text!r}")
        print(f"OUTPUT : {decoded!r}")
        sys.stdout.flush()

    model.train()


def print_comparison_table(logs_hgdm, logs_omega, params_hgdm, params_omega):
    print("\n")
    print("=" * 100)
    print("FINAL COMPARISON: HGDM (Previous)  vs  OmegaGDM (New)")
    print("=" * 100)

    # Aggregate metrics
    def stats(logs):
        losses = [l['loss'] for l in logs]
        bpbs   = [l['bpb'] for l in logs]
        vrams  = [l['vram_mb'] for l in logs if l['vram_mb'] >= 0]
        times  = [l['step_time'] for l in logs]
        return {
            "loss_start": losses[0] if losses else 0,
            "loss_final": losses[-1] if losses else 0,
            "loss_min":   min(losses) if losses else 0,
            "bpb_final":  bpbs[-1] if bpbs else 0,
            "vram_peak":  max(vrams) if vrams else 0,
            "vram_avg":   sum(vrams) / len(vrams) if vrams else 0,
            "time_per_step_avg": sum(times) / len(times) if times else 0,
        }

    h = stats(logs_hgdm)
    o = stats(logs_omega)

    def improvement(old, new, lower_is_better=True):
        if old == 0:
            return "N/A"
        pct = ((old - new) / abs(old)) * 100
        if lower_is_better:
            return f"{'+' if pct > 0 else ''}{pct:.1f}% {'better' if pct > 0 else 'worse'}"
        else:
            return f"{'+' if pct > 0 else ''}{pct:.1f}% {'better' if pct < 0 else 'worse'}"

    rows = [
        ("Parameters",         f"{params_hgdm/1e6:.2f}M",     f"{params_omega/1e6:.2f}M",   f"+{(params_omega-params_hgdm)/params_hgdm*100:.1f}% larger"),
        ("Starting Loss",      f"{h['loss_start']:.4f}",       f"{o['loss_start']:.4f}",     improvement(h['loss_start'], o['loss_start'])),
        ("Final Loss",         f"{h['loss_final']:.4f}",       f"{o['loss_final']:.4f}",     improvement(h['loss_final'], o['loss_final'])),
        ("Minimum Loss",       f"{h['loss_min']:.4f}",         f"{o['loss_min']:.4f}",       improvement(h['loss_min'], o['loss_min'])),
        ("Final BPB",          f"{h['bpb_final']:.4f}",        f"{o['bpb_final']:.4f}",      improvement(h['bpb_final'], o['bpb_final'])),
        ("Peak Net VRAM",      f"{h['vram_peak']}MB",          f"{o['vram_peak']}MB",        improvement(h['vram_peak'], o['vram_peak'])),
        ("Avg Net VRAM",       f"{h['vram_avg']:.0f}MB",       f"{o['vram_avg']:.0f}MB",     improvement(h['vram_avg'], o['vram_avg'])),
        ("Avg Step Time",      f"{h['time_per_step_avg']:.3f}s", f"{o['time_per_step_avg']:.3f}s", improvement(h['time_per_step_avg'], o['time_per_step_avg'])),
    ]

    print(f"\n{'Metric':<28} | {'HGDM (Previous)':<20} | {'OmegaGDM (New)':<20} | {'OmegaGDM Improvement'}")
    print("-" * 100)
    for metric, h_val, o_val, impr in rows:
        print(f"{metric:<28} | {h_val:<20} | {o_val:<20} | {impr}")
    print("=" * 100)

    # Save comparison to JSON
    with open("comparison_results.json", "w") as f:
        json.dump({
            "params_hgdm": params_hgdm,
            "params_omega": params_omega,
            "logs_hgdm": logs_hgdm,
            "logs_omega": logs_omega,
        }, f, indent=2)
    print("\n[System] Full comparison results saved to comparison_results.json")

def main():
    device = torch.device('cuda')
    assert torch.cuda.is_available(), "CUDA Environment Not Found."

    verify_datasets()

    # -------------------------------------------------------------------------
    # 1. MATCHING MODEL CONFIGURATIONS — scaled to 24GB GPU (~120M-140M params)
    # -------------------------------------------------------------------------
    config_hgdm = HGDMConfig(
        d_model=1024,
        n_layers=12,
        n_heads=16,
        d_k=64,
        d_v=64,
        d_ff=4096,
        max_position_embeddings=2048,
        vocab_size=256
    )

    config_omega = OmegaConfig(
        d_byte=256,
        catcher_layers=2,
        renderer_layers=2,
        d_model=1024,
        core_layers=12,
        n_heads=16,
        d_k=64,
        d_v=64,
        d_ff=4096,
        decimation_rate=8,
        max_position_embeddings=2048,
        vocab_size=256
    )

    print("\n================================================================")
    print("SEQUENTIAL COMPARISON TRAINING SPRINT: HGDM vs OmegaGDM")
    print("SCALE: ~120M-140M params | 24GB GPU | 2000 steps")
    print("================================================================")
    print("[Dataset] Mixture: 60% FineWeb-Edu, 25% Wikipedia, 15% Code")
    print(f"[Memory]  Initial Total VRAM Used (nvidia-smi): {get_net_gpu_memory()}MB")

    max_steps = 2000
    grad_accum_steps = 8
    batch_size = 4      # Scaled up: full GPU now free
    block_size = 2048   # Scaled up: full context window

    # -------------------------------------------------------------------------
    # 2. TRAIN HGDM (Previous) — 1000 steps on a fresh stream
    # -------------------------------------------------------------------------
    model_hgdm = HGDMUltimate(config_hgdm, force_sequential=False).to(device)
    params_hgdm = sum(p.numel() for p in model_hgdm.parameters())
    print(f"\n[HGDM]    Parameters: {params_hgdm/1e6:.3f} Million")

    logs_hgdm = run_training_sprint(
        model_hgdm, "HGDM (Previous)", max_steps,
        grad_accum_steps, batch_size, block_size, device
    )

    # Test generation immediately after HGDM finishes, before freeing GPU
    run_generation_test(model_hgdm, "HGDM (Previous)", device)

    # Free HGDM from GPU memory before OmegaGDM runs
    del model_hgdm
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 3. TRAIN OmegaGDM (New) — 1000 steps on a fresh stream (same data order)
    # -------------------------------------------------------------------------
    model_omega = OmegaGDM(config_omega, force_sequential=False).to(device)
    params_omega = sum(p.numel() for p in model_omega.parameters())
    print(f"\n[OmegaGDM] Parameters: {params_omega/1e6:.3f} Million")

    logs_omega = run_training_sprint(
        model_omega, "OmegaGDM (New)", max_steps,
        grad_accum_steps, batch_size, block_size, device
    )

    # Test generation immediately after OmegaGDM finishes, before freeing GPU
    run_generation_test(model_omega, "OmegaGDM (New)", device)

    del model_omega
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. PRINT FINAL COMPARISON TABLE
    # -------------------------------------------------------------------------
    print_comparison_table(logs_hgdm, logs_omega, params_hgdm, params_omega)

if __name__ == "__main__":
    main()
