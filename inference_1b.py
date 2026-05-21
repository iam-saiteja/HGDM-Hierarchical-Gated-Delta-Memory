import torch
import torch.nn.functional as F
import sys
import os
import math
import time
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hgdm_omega import OmegaGDM, OmegaConfig

# =============================================================================
# INFERENCE BENCHMARK — OmegaGDM 1B
# Tests: GPU generation, CPU generation, O(1) memory proof at varying lengths
# =============================================================================

PROMPTS = [
    "The theory of relativity states that",
    "In machine learning, a transformer model",
    "def fibonacci(n):\n    ",
    "The capital of France is Paris. The capital of Germany is",
    "Once upon a time, in a kingdom far away,",
    "The human brain contains approximately",
    "import torch\nimport torch.nn as nn\n\nclass ",
    "Water boils at 100 degrees Celsius. Ice melts at",
]

def get_gpu_memory():
    try:
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        return int(subprocess.check_output(cmd, shell=True).decode().strip())
    except:
        return -1

def get_config():
    return OmegaConfig(
        d_byte=256,
        catcher_layers=2,
        renderer_layers=2,
        d_model=2048,
        core_layers=18,
        n_heads=32,
        d_k=64,
        d_v=64,
        d_ff=5460,
        decimation_rate=8,
        max_position_embeddings=65536,
        vocab_size=256,
        use_state_fusion=False,
    )

def load_model(device, checkpoint_path="hgdm_1b_latest.pt"):
    config = get_config()
    model = OmegaGDM(config).to(device)

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"[System] Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    step = checkpoint.get('step', '?')
    tokens = checkpoint.get('tokens_trained', '?')
    print(f"[System] Loaded! Step {step} | {tokens:,} tokens trained")

    params = sum(p.numel() for p in model.parameters())
    print(f"[Model]  Parameters: {params/1e9:.3f} Billion")

    model.eval()
    return model

def generate_text(model, prompt_text, device, max_new_bytes=200, temp=0.7):
    prompt_bytes = list(prompt_text.encode('utf-8', errors='ignore'))
    x = torch.tensor([prompt_bytes], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(x, max_new_bytes=max_new_bytes, temp=temp)

    new_bytes = generated[0, len(prompt_bytes):].tolist()
    decoded = bytes(new_bytes).decode('utf-8', errors='replace')
    return decoded

# =============================================================================
# TEST 1: GPU Generation Suite
# =============================================================================
def test_gpu_generation(model, device):
    print(f"\n{'='*70}")
    print(f"  TEST 1: GPU GENERATION (device={device})")
    print(f"{'='*70}")

    vram_before = get_gpu_memory()
    print(f"  VRAM before generation: {vram_before} MB")

    for i, prompt in enumerate(PROMPTS):
        torch.cuda.synchronize()
        t0 = time.time()
        output = generate_text(model, prompt, device, max_new_bytes=200, temp=0.7)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        vram_during = get_gpu_memory()
        bytes_generated = len(output.encode('utf-8', errors='ignore'))

        print(f"\n{'─'*70}")
        print(f"  Prompt {i+1}  |  Time: {elapsed:.2f}s  |  VRAM: {vram_during} MB  |  Bytes: {bytes_generated}")
        print(f"  INPUT  : {prompt!r}")
        print(f"  OUTPUT : {output!r}")

    vram_after = get_gpu_memory()
    print(f"\n  VRAM after all generations: {vram_after} MB")

# =============================================================================
# TEST 2: O(1) Memory Per Step Proof — Varying Sequence Lengths
# =============================================================================
def test_o1_memory_scaling(model, device):
    print(f"\n{'='*70}")
    print(f"  TEST 2: O(1) MEMORY SCALING PROOF")
    print(f"  Generating at varying prompt AND output lengths.")
    print(f"  If VRAM stays constant, memory is O(1) per generation step.")
    print(f"{'='*70}")

    base_text = "The quick brown fox jumps over the lazy dog. "
    prompt_lengths = [64, 128, 256, 512, 1024]
    gen_lengths = [50, 100, 200, 500]

    print(f"\n  {'Prompt Len':<12} | {'Gen Len':<9} | {'Gen Time':<10} | {'VRAM':<10} | {'Bytes/sec':<10}")
    print(f"  {'-'*60}")

    for target_len in prompt_lengths:
        for gen_len in gen_lengths:
            repeats = max(1, target_len // len(base_text.encode('utf-8')))
            prompt = (base_text * repeats)[:target_len]
            prompt_bytes = list(prompt.encode('utf-8', errors='ignore'))
            x = torch.tensor([prompt_bytes], dtype=torch.long, device=device)

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            t0 = time.time()
            with torch.no_grad():
                generated = model.generate(x, max_new_bytes=gen_len, temp=0.7)
            torch.cuda.synchronize()
            elapsed = time.time() - t0

            vram_post = get_gpu_memory()
            bytes_per_sec = gen_len / elapsed if elapsed > 0 else 0

            print(f"  {len(prompt_bytes):<12} | {gen_len:<9} | {elapsed:<10.2f}s | {vram_post:<10} MB | {bytes_per_sec:<10.1f}")
        print()  # blank line between prompt groups

    print(f"  KEY: If VRAM column stays ~constant across ALL rows,")
    print(f"  the architecture achieves O(1) memory per generation step.")
    print(f"  If Bytes/sec stays ~constant across prompt lengths,")
    print(f"  inference cost is independent of context length.")

# =============================================================================
# MAIN
# =============================================================================
def main():
    device = torch.device('cuda')
    assert torch.cuda.is_available(), "CUDA required."
    print(f"[System] Using device: {device}")

    model = load_model(device)

    # TEST 1: GPU generation with VRAM tracking
    test_gpu_generation(model, device)

    # TEST 2: O(1) memory proof
    test_o1_memory_scaling(model, device)

    print(f"\n{'='*70}")
    print(f"  ALL INFERENCE TESTS COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
