import sys
import os
import torch
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tokenized.config import TokenizedConfig

def run_benchmark():
    cfg = TokenizedConfig.get_tier('pilot')
    hgdm_state_bytes = cfg.n_layers * cfg.n_heads * cfg.d_k * cfg.d_v * 2  # bfloat16
    
    print("\n" + "=" * 78)
    print("      THEORETICAL & MEASURED MEMORY FOOTPRINT PER CONCURRENT USER")
    print("=" * 78)
    print(f"Model Configuration: {cfg.n_layers} Layers, {cfg.n_heads} Heads, d_k={cfg.d_k}, d_v={cfg.d_v}")
    print(f"Fixed HGDM State Size: {hgdm_state_bytes / 1024:.2f} KB PER USER (Constant forever)")
    print("-" * 78)
    print(f"{'Context (T)':>12} | {'Transformer KV':>16} | {'HGDM State':>14} | {'Memory Savings Ratio':>22}")
    print("-" * 78)
    
    for T in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        trans_kv_bytes = 2 * cfg.n_layers * cfg.n_heads * cfg.d_k * T * 2
        ratio = trans_kv_bytes / hgdm_state_bytes
        print(f"{T:12,d} | {trans_kv_bytes/(1024*1024):13.2f} MB | {hgdm_state_bytes/(1024*1024):11.3f} MB | {ratio:20.1f}x smaller")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    run_benchmark()
