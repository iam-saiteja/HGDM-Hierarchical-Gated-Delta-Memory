import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from hgdm_ultimate import HGDMUltimate, HGDMConfig

# =============================================================================
# 1. EVALUATION CONFIG
# =============================================================================
CHECKPOINT = "math_latest.pt"
SEQ_LEN = 2048

def evaluate_math():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 TITAN-MATH-1B: INFERENCE EVALUATOR")
    print(f"Loading checkpoint from: {CHECKPOINT}...")

    # Load Model
    config = HGDMConfig(d_model=1792, n_layers=20, n_heads=28, d_ff=7168)
    model = HGDMUltimate(config).to(device)
    
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded Brain at Step {ckpt['step']} ({ckpt['total_tokens']/1e6:.1f}M tokens)")
    else:
        print("!!! No checkpoint found. Exiting.")
        return

    model.eval()

    while True:
        prompt = input("\n[MATH PROMPT] (e.g. 'Solve for x: 2x + 5 = 15') -> ")
        if prompt.lower() in ['exit', 'quit']: break
        
        # Byte Encoding
        input_bytes = list(prompt.encode('utf-8'))
        x = torch.tensor([input_bytes], dtype=torch.long).to(device)
        
        print("\n[TITAN REASONING]: ", end="", flush=True)
        
        # Generation Loop
        generated = []
        with torch.no_grad():
            for _ in range(512): # Max tokens
                with torch.amp.autocast('cuda'):
                    logits, _ = model(x)
                
                # Sample next token (Low temp for logic)
                next_token_logits = logits[0, -1, :] / 0.2 
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated.append(next_token.item())
                
                # Stream the byte output
                try:
                    char = bytes([next_token.item()]).decode('utf-8')
                    print(char, end="", flush=True)
                except:
                    print(".", end="", flush=True)
                
                x = torch.cat([x, next_token.unsqueeze(0)], dim=1)
                if x.size(1) > SEQ_LEN: x = x[:, 1:]
                
                if next_token.item() == 0: break # End of sequence
        print("\n")

if __name__ == "__main__":
    evaluate_math()
