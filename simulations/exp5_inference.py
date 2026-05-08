import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import json
from hgdm_ultimate import HGDMUltimate, HGDMConfig

def run_long_inference():
    device = torch.device('cuda')
    print("\n--- Testing Long-Sequence Inference (2000 bytes) ---")
    
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    model = HGDMUltimate(config).to(device)
    
    checkpoint_path = "hgdm_enwik8_120M.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading trained checkpoint from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    else:
        print(f"WARNING: {checkpoint_path} not found. Running with untrained model.")
        print("Please run exp1_enwik8_main.py first to generate the checkpoint.")
        
    model.eval()
    
    prompt = torch.tensor([list("Wikipedia is ".encode('utf-8'))], dtype=torch.long, device=device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Using the built-in generate method from hgdm_ultimate.py
            output_bytes = model.generate(prompt, max_new_bytes=2000, temp=0.8)[0]
            
    text = bytes(output_bytes.tolist()).decode('utf-8', errors='replace')
    
    print("\nGeneration successful!")
    print("Length of generated sequence:", len(output_bytes))
    print("\nSample Output:\n" + text[:500] + "\n... [truncated]")
    
    results = {
        "prompt": "Wikipedia is ",
        "generated_length": len(output_bytes),
        "text_sample": text
    }
    
    with open("results_exp5.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nExperiment 5 Complete. Saved results_exp5.json")

if __name__ == "__main__":
    run_long_inference()
