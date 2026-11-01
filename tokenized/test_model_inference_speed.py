import time
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenized.config import TokenizedConfig
from tokenized.model_hgdm import TokenizedHGDM
from transformers import AutoTokenizer

def benchmark_inference():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")
    ckpt_path = "checkpoints_tokenized/ckpt_hgdm_sft.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints_tokenized/ckpt_hgdm.pt"
        
    print(f"Loading checkpoint: {ckpt_path}")
    data = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = data["config"]
    model = TokenizedHGDM(cfg).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "User: Explain how the brain processes language.\nAssistant:"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = model.generate(prompt_ids=prompt_ids, max_new_tokens=20)

    # Benchmark 200 tokens
    N_TOKENS = 200
    print(f"Generating {N_TOKENS} tokens with Fused Nitro Step Kernel...")
    torch.cuda.synchronize()
    t0 = time.time()
    
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_ids = model.generate(prompt_ids=prompt_ids, max_new_tokens=N_TOKENS, temperature=0.7)
            
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    new_tokens = output_ids.shape[1] - prompt_ids.shape[1]
    speed = new_tokens / elapsed

    text = tokenizer.decode(output_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)
    print("\n" + "=" * 65)
    print(f"GENERATION COMPLETED in {elapsed:.3f} seconds")
    print(f"Throughput: {speed:.1f} tokens/second")
    print("=" * 65)
    print(f"Sample output: {text[:200]}...")

if __name__ == "__main__":
    benchmark_inference()
