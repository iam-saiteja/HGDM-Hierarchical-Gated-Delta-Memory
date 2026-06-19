import torch
import torch.nn.functional as F
import math
import os
import sys
import argparse
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hgdm_omega import OmegaGDM, OmegaConfig

def get_wikitext_bytes():
    print("[Benchmark] Loading wikitext-2-raw-v1 (test split)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(dataset['text'])
    byte_seq = list(text.encode('utf-8', errors='replace'))
    print(f"[Benchmark] Loaded {len(byte_seq):,} bytes for evaluation.")
    return byte_seq

def benchmark_omega():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="omega_v1_dpo_latest.pt", help="Checkpoint to evaluate")
    parser.add_argument("--block-size", type=int, default=2048, help="Context window size")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cfg = OmegaConfig(
        d_byte=256, catcher_layers=2, renderer_layers=2,
        d_model=768, core_layers=12, n_heads=12,
        d_k=64, d_v=64, d_ff=3072,
        decimation_rate=8, max_position_embeddings=2048,
        vocab_size=256, use_state_fusion=False
    )
    
    print(f"[Benchmark] Initializing OmegaGDM on {device}...")
    model = OmegaGDM(cfg, force_sequential=False).to(device)
    
    if not os.path.exists(args.ckpt):
        print(f"[Error] Checkpoint not found: {args.ckpt}")
        sys.exit(1)
        
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    print("[Benchmark] Model loaded successfully.")
    
    data_bytes = get_wikitext_bytes()
    total_loss = 0.0
    total_bytes = 0
    
    chunk_size = args.block_size
    num_chunks = len(data_bytes) // chunk_size
    
    print(f"[Benchmark] Evaluating {num_chunks} chunks of size {chunk_size}...")
    
    with torch.no_grad():
        for i in tqdm(range(num_chunks)):
            chunk = data_bytes[i*chunk_size : (i+1)*chunk_size + 1]
            if len(chunk) <= chunk_size:
                break
                
            x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
            
            logits, _ = model(x)
            
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1), reduction='sum')
            total_loss += loss.item()
            total_bytes += chunk_size
            
    avg_loss = total_loss / total_bytes
    bpb = avg_loss / math.log(2)
    perplexity = math.exp(avg_loss)
    
    print("\n=========================================")
    print("[BENCHMARK RESULTS - WIKITEXT-2 ZERO-SHOT]")
    print("=========================================")
    print(f"Total Bytes Evaluated: {total_bytes:,}")
    print(f"Cross Entropy Loss:    {avg_loss:.4f}")
    print(f"Bits-Per-Byte (BPB):   {bpb:.4f}")
    print(f"Perplexity:            {perplexity:.4f}")
    print("=========================================\n")

if __name__ == "__main__":
    benchmark_omega()
