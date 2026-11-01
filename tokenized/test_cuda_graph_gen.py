import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenized.config import TokenizedConfig
from tokenized.model_hgdm import TokenizedHGDM
from tokenized.cuda_graph import CUDAGraphedHGDM
from transformers import AutoTokenizer


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
        
    device = torch.device("cuda")
    ckpt_path = "checkpoints_tokenized/ckpt_hgdm_sft.pt"
    data = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = data["config"]
    model = TokenizedHGDM(cfg).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "User: What is artificial intelligence and how does it work?\nAssistant:"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    N_TOKENS = 256
    
    # 1. Benchmark Standard Eager Autoregressive Generation
    print("\n[1/2] Benchmarking Standard Eager Generation (with TSG step kernel)...")
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Warmup
            _ = model.generate(prompt_ids, max_new_tokens=20)
            torch.cuda.synchronize()
            t0 = time.time()
            eager_out = model.generate(prompt_ids, max_new_tokens=N_TOKENS)
            torch.cuda.synchronize()
            t_eager = time.time() - t0
            eager_toks = eager_out.shape[1] - prompt_ids.shape[1]
            eager_speed = eager_toks / t_eager

    # 2. Benchmark Fused GEMM + CUDA Graphs
    print("\n[2/2] Benchmarking Fused GEMM + CUDA Graph Pulse Engine...")
    model.fuse_projections()
    graphed_model = model.to_cuda_graph(device)
    
    # Warmup
    _ = graphed_model.generate(prompt_ids, max_new_tokens=20)
    
    torch.cuda.synchronize()
    t0 = time.time()
    graphed_out = graphed_model.generate(prompt_ids, max_new_tokens=N_TOKENS)
    torch.cuda.synchronize()
    t_graphed = time.time() - t0
    
    graphed_toks = graphed_out.shape[1] - prompt_ids.shape[1]
    graphed_speed = graphed_toks / t_graphed
    speedup = graphed_speed / eager_speed
    
    resp = tokenizer.decode(graphed_out[0, prompt_ids.shape[1]:], skip_special_tokens=True)
    
    print("\n" + "=" * 75)
    print("      FUSED GEMM + CUDA GRAPH ACCELERATION BENCHMARK (RTX 3090 Ti)")
    print("=" * 75)
    print(f"Eager TSG Generation:       {eager_toks} tokens in {t_eager:.4f}s  --> {eager_speed:6.1f} tok/s")
    print(f"Fused GEMM + CUDA Graph:    {graphed_toks} tokens in {t_graphed:.4f}s  --> {graphed_speed:6.1f} tok/s")
    print(f"Hardware Acceleration:      {speedup:4.2f}x SPEEDUP ({graphed_speed - eager_speed:+.1f} tok/s)")
    print("=" * 75)
    print(f"Response Preview:\n{resp[:250]}...\n")

if __name__ == "__main__":
    main()
