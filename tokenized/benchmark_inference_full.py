import time
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenized.config import TokenizedConfig
from tokenized.model_hgdm import TokenizedHGDM
from tokenized.model_transformer import MatchedTransformer
from tokenized.cuda_graph import CUDAGraphedHGDM


def get_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def benchmark_model_generation(model, prompt_ids, max_new_tokens=128, is_cuda_graph=False):
    device = prompt_ids.device
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.time()
    if is_cuda_graph:
        output_ids = model.generate(prompt_ids=prompt_ids, max_new_tokens=max_new_tokens)
    else:
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7
                )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = max(1e-4, time.time() - t0)

    n_new = output_ids.shape[1] - prompt_ids.shape[1]
    tok_per_sec = n_new / elapsed
    ms_per_tok = (elapsed / n_new) * 1000.0
    peak_vram = get_vram_mb()

    return output_ids, tok_per_sec, ms_per_tok, peak_vram, elapsed


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")
    print("=" * 80)
    print("      COMPREHENSIVE INFERENCE BENCHMARK: HGDM-HBS vs. TRANSFORMER")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Load HGDM-HBS Model
    hgdm_ckpt = "checkpoints_tokenized/full_hbs/ckpt_hgdm_sft.pt"
    if not os.path.exists(hgdm_ckpt):
        hgdm_ckpt = "checkpoints_tokenized/full_hbs/ckpt_hgdm.pt"
    print(f"\n[1/3] Loading HGDM-HBS Checkpoint: {hgdm_ckpt}")
    data_h = torch.load(hgdm_ckpt, map_location=device, weights_only=False)
    cfg_h = data_h["config"]
    model_h = TokenizedHGDM(cfg_h).to(device)
    model_h.load_state_dict(data_h["model_state"])
    model_h.eval()

    # Create CUDA Graphed Version
    print("[CUDA Graph] Capturing Fused GEMM + CUDA Graph engine...")
    model_h_graph = TokenizedHGDM(cfg_h).to(device)
    model_h_graph.load_state_dict(data_h["model_state"])
    model_h_graph.eval()
    model_h_graph.fuse_projections()
    cuda_graph_engine = model_h_graph.to_cuda_graph(device)

    # 2. Load Matched Transformer Model
    trans_ckpt = "checkpoints_tokenized/ckpt_transformer_sft.pt"
    if not os.path.exists(trans_ckpt):
        trans_ckpt = "checkpoints_tokenized/ckpt_transformer.pt"
    print(f"[2/3] Loading Matched Transformer Checkpoint: {trans_ckpt}")
    data_t = torch.load(trans_ckpt, map_location=device, weights_only=False)
    cfg_t = data_t["config"]
    model_t = MatchedTransformer(cfg_t).to(device)
    model_t.load_state_dict(data_t["model_state"])
    model_t.eval()

    # 3. Benchmark 1: Speed & Latency (256 Tokens Generation)
    print("\n" + "=" * 80)
    print("BENCHMARK 1: DECODING SPEED, LATENCY & VRAM (256 Tokens Generated)")
    print("=" * 80)

    test_prompt = "User: Explain the core principles of artificial intelligence and machine learning.\nAssistant:"
    prompt_ids = torch.tensor([tokenizer.encode(test_prompt)], device=device)
    print(f"Prompt: \"{test_prompt}\" ({prompt_ids.shape[1]} tokens)")

    # Warmup
    _ = model_t.generate(prompt_ids, max_new_tokens=15)
    _ = model_h.generate(prompt_ids, max_new_tokens=15)
    _ = cuda_graph_engine.generate(prompt_ids, max_new_tokens=15)

    # Run Transformer
    out_t, speed_t, lat_t, vram_t, time_t = benchmark_model_generation(model_t, prompt_ids, max_new_tokens=256)
    # Run HGDM Eager
    out_he, speed_he, lat_he, vram_he, time_he = benchmark_model_generation(model_h, prompt_ids, max_new_tokens=256)
    # Run HGDM CUDA Graph
    out_hg, speed_hg, lat_hg, vram_hg, time_hg = benchmark_model_generation(cuda_graph_engine, prompt_ids, max_new_tokens=256, is_cuda_graph=True)

    print(f"\n{'Engine':<32} | {'Throughput':>12} | {'Latency/Token':>14} | {'Peak VRAM':>10}")
    print("-" * 76)
    print(f"{'Matched Transformer (KV-Cache)':<32} | {speed_t:10.1f} tok/s | {lat_t:11.2f} ms | {vram_t:8.1f} MB")
    print(f"{'HGDM-HBS (Eager)':<32} | {speed_he:10.1f} tok/s | {lat_he:11.2f} ms | {vram_he:8.1f} MB")
    print(f"{'HGDM-HBS (Fused + CUDA Graph)':<32} | {speed_hg:10.1f} tok/s | {lat_hg:11.2f} ms | {vram_hg:8.1f} MB")
    print("-" * 76)
    print(f"CUDA Graph Acceleration vs Eager: {speed_hg / max(1e-4, speed_he):.2f}x speedup")

    # 4. Benchmark 2: Memory Footprint vs. Context Length (Scaling)
    print("\n" + "=" * 80)
    print("BENCHMARK 2: STATE / KV-CACHE MEMORY SCALING VS CONTEXT LENGTH")
    print("=" * 80)
    # HGDM-HBS state size = n_bands * n_layers * n_heads * d_k * d_v * 2 bytes
    n_bands = getattr(cfg_h, "hbs_n_bands", 4)
    hgdm_fixed_bytes = n_bands * cfg_h.n_layers * cfg_h.n_heads * cfg_h.d_k * cfg_h.d_v * 2
    hgdm_fixed_kb = hgdm_fixed_bytes / 1024.0

    print(f"Fixed HGDM-HBS Recurrent State Size: {hgdm_fixed_kb:.1f} KB (CONSTANT forever)")
    print(f"{'Context (Tokens)':>16} | {'Transformer KV-Cache':>22} | {'HGDM State':>14} | {'Memory Savings':>16}")
    print("-" * 76)
    for T in [128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
        trans_kv_kb = (2 * cfg_t.n_layers * cfg_t.n_heads * cfg_t.d_k * T * 2) / 1024.0
        ratio = trans_kv_kb / hgdm_fixed_kb
        print(f"{T:16,d} | {trans_kv_kb:19.1f} KB | {hgdm_fixed_kb:11.1f} KB | {ratio:14.1f}x")

    # 5. Benchmark 3: Qualitative Response Comparison
    print("\n" + "=" * 80)
    print("BENCHMARK 3: GENERATION SAMPLES (SIDE-BY-SIDE)")
    print("=" * 80)
    prompts = [
        "User: What is machine learning in simple terms?\nAssistant:",
        "User: Explain why computers use binary code.\nAssistant:",
        "User: Write a short explanation of how gravity works.\nAssistant:"
    ]

    for i, p_text in enumerate(prompts, 1):
        p_ids = torch.tensor([tokenizer.encode(p_text)], device=device)
        print(f"\n--- PROMPT {i}: \"{p_text.splitlines()[0]}\" ---")
        
        # Transformer
        resp_t_ids = model_t.generate(p_ids, max_new_tokens=64, temperature=0.7)
        resp_t = tokenizer.decode(resp_t_ids[0, p_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # HGDM CUDA Graph
        resp_h_ids = cuda_graph_engine.generate(p_ids, max_new_tokens=64)
        resp_h = tokenizer.decode(resp_h_ids[0, p_ids.shape[1]:], skip_special_tokens=True).strip()

        print(f"\n[TRANSFORMER RESPONSE]:\n{resp_t}\n")
        print(f"[HGDM-HBS RESPONSE (CUDA GRAPH)]:\n{resp_h}\n")
        print("-" * 80)


if __name__ == "__main__":
    main()
