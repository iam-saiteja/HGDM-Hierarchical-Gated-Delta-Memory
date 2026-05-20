import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import subprocess
import math
import json
import sys
import argparse
from hgdm_omega import OmegaGDM, OmegaConfig
from data_1b import get_1b_dataloader

# =============================================================================
# OMEGAGDM V2 — TRAINING + INFERENCE
# Config kept identical to the HGDM baseline run for fair comparison.
# =============================================================================

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
        print("[Dataset] All streaming pipelines verified!")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Dataset verification failed: {e}")
        sys.exit(1)

PROMPTS = [
    "The theory of relativity states that",
    "In machine learning, a transformer model",
    "def fibonacci(n):\n    ",
    "The capital of France is Paris. The capital of Germany is",
]

def run_generation_test(model, device, max_new_bytes=150, temp=0.8):
    model.eval()
    print(f"\n{'='*60}")
    print("GENERATION TEST: OmegaGDM V2")
    print(f"{'='*60}")
    for i, prompt_text in enumerate(PROMPTS):
        prompt_bytes = list(prompt_text.encode('utf-8', errors='ignore'))
        prompt_tensor = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
        try:
            with torch.no_grad():
                generated = model.generate(prompt_tensor, max_new_bytes=max_new_bytes, temp=temp)
            new_bytes = generated[0, len(prompt_bytes):].tolist()
            decoded = bytes(new_bytes).decode('utf-8', errors='replace')
        except Exception as e:
            decoded = f"[ERROR: {e}]"
        print(f"\n--- Prompt {i+1} ---")
        print(f"PROMPT : {prompt_text!r}")
        print(f"OUTPUT : {decoded!r}")
        sys.stdout.flush()
    model.train()

def main():
    parser = argparse.ArgumentParser(description="Train OmegaGDM V2")
    parser.add_argument("--no-precheck", action="store_true", help="Skip dataset streaming pre-verification")
    args = parser.parse_args()

    device = torch.device('cuda')
    assert torch.cuda.is_available(), "CUDA not found."

    if not args.no_precheck:
        verify_datasets()
    else:
        print("[Dataset] Skipping dataset pre-verification check as requested.")

    # -------------------------------------------------------------------------
    # MODEL CONFIG — identical to HGDM baseline for fair comparison
    # -------------------------------------------------------------------------
    config = OmegaConfig(
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
        vocab_size=256,
        use_state_fusion=False,     # set True to activate CrossLayerStateFusion in core
    )

    model = OmegaGDM(config, force_sequential=False).to(device)
    params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"OmegaGDM V2 — Training Run")
    print(f"{'='*60}")
    print(f"[Model]  Parameters:   {params/1e6:.3f} Million")
    print(f"[Memory] Initial VRAM: {get_gpu_memory()}MB")
    print(f"[Data]   Mixture: 60% FineWeb-Edu | 25% Wikipedia | 15% Code")

    # -------------------------------------------------------------------------
    # TRAINING HYPERPARAMETERS — identical to HGDM baseline
    # -------------------------------------------------------------------------
    max_steps       = 500
    grad_accum      = 4
    batch_size      = 8
    block_size      = 2048
    lr              = 4e-4
    weight_decay    = 0.01
    grad_clip       = 1.0
    log_every       = 25
    log_file        = "omega_v2_train_logs.jsonl"

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=1e-5)

    dataloader  = get_1b_dataloader(block_size=block_size, batch_size=batch_size)
    data_stream = iter(dataloader)

    model.train()
    logs     = []
    t_start  = time.time()

    print(f"\n{'Step':<5} | {'Loss':<10} | {'BPB':<7} | {'VRAM':<8} | {'StepTime':<9} | {'Elapsed'}")
    print("-" * 65)
    sys.stdout.flush()

    vram_cache = get_gpu_memory()

    for step in range(max_steps):
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        t_step = time.time()

        for _ in range(grad_accum):
            batch = next(data_stream).to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, 256), y.reshape(-1)
                ) / grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[ERROR] NaN/Inf loss at step {step}. Stopping.")
                break

            loss.backward()
            accum_loss += loss.item() * grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()

        step_time = time.time() - t_step
        bpb = accum_loss / math.log(2)

        if step % 5 == 0:
            vram_cache = get_gpu_memory()

        logs.append({
            "step": step,
            "loss": accum_loss,
            "bpb": bpb,
            "vram_mb": vram_cache,
            "step_time": step_time,
        })

        if step % log_every == 0 or step == max_steps - 1:
            elapsed = (time.time() - t_start) / 60
            temp = get_gpu_temp()
            print(
                f"{step:04d} | {accum_loss:<10.4f} | {bpb:<7.4f} | "
                f"{vram_cache:<5}MB | {step_time:.2f}s     | {elapsed:.1f}min  ({temp})"
            )
            sys.stdout.flush()

    # -------------------------------------------------------------------------
    # SAVE LOGS
    # -------------------------------------------------------------------------
    with open(log_file, "w") as f:
        for entry in logs:
            f.write(json.dumps(entry) + "\n")
    print(f"\n[System] Training complete. Logs saved to {log_file}")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    losses    = [l['loss'] for l in logs]
    vrams     = [l['vram_mb'] for l in logs if l['vram_mb'] >= 0]
    times     = [l['step_time'] for l in logs]
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY — OmegaGDM V2")
    print(f"{'='*60}")
    print(f"  Parameters:      {params/1e6:.3f}M")
    print(f"  Starting Loss:   {losses[0]:.4f}")
    print(f"  Final Loss:      {losses[-1]:.4f}")
    print(f"  Minimum Loss:    {min(losses):.4f}")
    print(f"  Final BPB:       {losses[-1]/math.log(2):.4f}")
    print(f"  Peak VRAM:       {max(vrams)}MB")
    print(f"  Avg Step Time:   {sum(times)/len(times):.3f}s")
    print(f"  Total Time:      {(time.time()-t_start)/60:.1f} min")
    print(f"{'='*60}")

    # -------------------------------------------------------------------------
    # GENERATION TEST
    # -------------------------------------------------------------------------
    run_generation_test(model, device)

if __name__ == "__main__":
    main()
