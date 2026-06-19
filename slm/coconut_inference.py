import torch
import argparse
import sys
import os

# Ensure local import paths work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hgdm_omega import OmegaGDM, OmegaConfig

def decode_byte(b):
    """Safely decode a byte to a string, escaping unprintables."""
    try:
        char = bytes([b]).decode('utf-8')
        if char.isprintable() or char in ['\n', '\t']:
            return char
        else:
            return f"\\x{b:02x}"
    except UnicodeDecodeError:
        return f"\\x{b:02x}"

def main():
    parser = argparse.ArgumentParser(description="OmegaGDM True Coconut Latent Thought Inference")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint (e.g. omega_v1_dpo_latest.pt)')
    parser.add_argument('--thought_steps', type=int, default=5, help='Number of continuous latent thought steps before answering')
    parser.add_argument('--max_new_bytes', type=int, default=100, help='Maximum real bytes to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature for the final output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Loading OmegaGDM 120M from {args.ckpt}...")
    
    # Using the 120M configuration
    config = OmegaConfig(
        d_byte=256, catcher_layers=2, renderer_layers=2, 
        d_model=768, core_layers=12, n_heads=12, 
        d_k=64, d_v=64, d_ff=3072, decimation_rate=8, 
        max_position_embeddings=512, vocab_size=256, use_state_fusion=False
    )
    
    model = OmegaGDM(config, force_sequential=False).to(device)
    
    if os.path.exists(args.ckpt):
        try:
            ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
            print("[*] Weights loaded successfully.")
        except Exception as e:
            print(f"[!] Error loading weights: {e}")
            print(f"[!] Running with untrained weights.")
    else:
        print(f"[!] Warning: Checkpoint {args.ckpt} not found! Running with untrained weights.")
        
    model.eval()
    model.bfloat16()  # Force all weights to bfloat16 to prevent Index Put dtype mismatches

    print("\n" + "="*60)
    print("   TRUE LATENT COCONUT THINKING INTERFACE")
    print("   Translate to English Toggle: [ON]")
    print("="*60)

    while True:
        try:
            prompt = input("\nUser: ")
            if not prompt.strip():
                continue
            if prompt.lower() in ['quit', 'exit']:
                break
        except (EOFError, KeyboardInterrupt):
            break

        print("OmegaGDM: ", end='', flush=True)
        
        prompt_bytes = list(prompt.encode('utf-8', errors='ignore'))
        # Optional: Add a special token or newline if your DPO required it
        # prompt_bytes += [ord('\n')]
        
        x = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
        
        # 1. Process the context completely
        states = None
        offset = 0
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # We need return_latent=True to get the continuous semantic vector (x_out)
                logits, states, x_out = model(
                    byte_seq=x, 
                    states=states, 
                    offset=offset, 
                    return_latent=True
                )
                offset += x.shape[1]
                
                # Extract the last continuous vector to begin Latent Thought
                latent_vector = x_out[:, -1:, :]
                
                # ========================================================
                # 2. LATENT COCONUT THINKING PHASE
                # ========================================================
                # We do NOT pass discrete bytes. We pass the continuous 
                # semantic vector directly back into the ODE engine!
                
                print("\033[90m[Thinking: \033[0m", end='', flush=True)
                for _ in range(args.thought_steps):
                    logits, states, new_x_out = model(
                        x_embed=latent_vector,  # True Continuous Thought!
                        states=states,
                        offset=offset,
                        return_latent=True
                    )
                    
                    # Translate to English Toggle
                    # We peek at the hidden thought by decoding the logits
                    thought_byte = torch.argmax(logits[:, -1, :], dim=-1).item()
                    char = decode_byte(thought_byte)
                    print(f"\033[90m{char}\033[0m", end='', flush=True)
                    
                    # Feed the new continuous semantic state forward
                    latent_vector = new_x_out[:, -1:, :]
                    offset += 1
                
                print("\033[90m] \033[0m", end='', flush=True)
                
                # ========================================================
                # 3. ACTUAL AUTOREGRESSIVE GENERATION
                # ========================================================
                # The final thought logit is sampled to produce the first real byte
                next_byte = torch.argmax(logits[:, -1, :], dim=-1).item()
                next_char = decode_byte(next_byte)
                print(next_char, end='', flush=True)
                
                current_byte = torch.tensor([[next_byte]], dtype=torch.long, device=device)
                
                for _ in range(args.max_new_bytes):
                    logits, states = model(byte_seq=current_byte, states=states, offset=offset)
                    
                    # Apply temperature scaling
                    scaled_logits = logits[:, -1, :] / args.temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    next_byte = torch.multinomial(probs, num_samples=1).item()
                    
                    char = decode_byte(next_byte)
                    print(char, end='', flush=True)
                    
                    current_byte = torch.tensor([[next_byte]], dtype=torch.long, device=device)
                    offset += 1
        print() # Newline after generation

if __name__ == "__main__":
    main()
