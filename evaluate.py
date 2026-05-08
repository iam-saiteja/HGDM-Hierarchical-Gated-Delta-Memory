import torch
import time
import os
import subprocess
from hgdm_ultimate import HGDMUltimate, HGDMConfig

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================
CHECKPOINT_PATH = "titan_latest.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    "The Roman Empire was ",
    "Photosynthesis is the process ",
    "The law of gravity states ",
    "Quantum mechanics is a ",
    "Artificial intelligence can be "
]

def get_gpu_stats():
    try:
        cmd = "nvidia-smi --query-gpu=temperature.gpu,memory.used --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        temp, mem = output.split(',')
        return temp.strip(), mem.strip()
    except:
        return "N/A", "N/A"

def generate(model, prompt, max_new_tokens=5000, temperature=0.7):
    model.eval()
    input_ids = torch.tensor([list(prompt.encode('utf-8'))], dtype=torch.long).to(DEVICE)
    
    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(input_ids, max_new_bytes=max_new_tokens, temp=temperature)
    end_time = time.time()
    
    total_time = end_time - start_time
    tokens_per_sec = max_new_tokens / total_time
    
    # Decode and Clean (Byte-Level Recovery)
    raw_bytes = bytes(generated_ids[0].cpu().tolist())
    output_text = raw_bytes.decode('utf-8', errors='ignore')
    return output_text, total_time, tokens_per_sec

def run_evaluation():
    print("="*60)
    print("🚀 TITAN-1B INFERENCE EVALUATOR")
    print("="*60)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: {CHECKPOINT_PATH} not found!")
        return

    # 1. Load Architecture (Titan-1B Scale)
    config = HGDMConfig(
        d_model=1792,
        n_layers=20,
        n_heads=28,
        d_k=64,
        d_v=64,
        d_ff=7168,
        vocab_size=256
    )
    model = HGDMUltimate(config).to(DEVICE)
    
    # 2. Load Weights (Metadata Aware)
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print(f">>> Found Titan Mission Metadata (Step {checkpoint.get('step', 'N/A')})")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Backward compatibility for raw state_dicts
        model.load_state_dict(checkpoint)
    
    del checkpoint
    torch.cuda.empty_cache()
    print("Model loaded successfully.")
    
    temp_init, mem_init = get_gpu_stats()
    print(f"Baseline GPU: {temp_init}C | {mem_init}MB VRAM")
    print("-" * 60)

    for i, prompt in enumerate(PROMPTS):
        print(f"\n[Test {i+1}] Prompt: '{prompt}'")
        
        text, duration, speed = generate(model, prompt)
        temp_curr, mem_curr = get_gpu_stats()
        
        print(f"\nResponse:\n{text}")
        print("-" * 40)
        print(f"Speed: {speed:.2f} bytes/sec | Time: {duration:.2f}s")
        print(f"GPU State: {temp_curr}C | {mem_curr}MB VRAM")
        print("-" * 60)

if __name__ == "__main__":
    run_evaluation()
