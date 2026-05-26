import torch
import torch.nn.functional as F
import sys
import os

# Add current path for module resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hgdm_omega import OmegaGDM, OmegaConfig

def get_config():
    # Matches the configuration used during training
    return OmegaConfig(
        d_byte=256,
        catcher_layers=1,
        renderer_layers=1,
        d_model=512,
        core_layers=8,
        n_heads=8,
        d_k=64,
        d_v=64,
        d_ff=2048,
        decimation_rate=8,
        max_position_embeddings=2048,
        vocab_size=256,
        use_state_fusion=False
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="HGDM English-to-Hindi Causal Translator CLI")
    parser.add_argument("--ckpt", default="hgdm_translation_latest.pt", help="Path to checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[System] Loading model on: {device}")

    # Load configuration and model architecture
    config = get_config()
    model = OmegaGDM(config, force_sequential=False).to(device)

    # Load model parameters
    if not os.path.exists(args.ckpt):
        print(f"[ERROR] Checkpoint not found: {args.ckpt}. Make sure to train the model first.")
        sys.exit(1)

    print(f"[System] Loading checkpoint from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[System] Loaded checkpoint from training step {checkpoint.get('step', '?')}")
    model.eval()

    print("\n" + "="*60)
    print("  HGDM ENGLISH-TO-HINDI CAUSAL TRANSLATION CLI")
    print("  Type 'exit' or 'quit' to close.")
    print("="*60 + "\n")

    while True:
        try:
            english_input = input("English: ").strip()
            if not english_input:
                continue
            if english_input.lower() in ['exit', 'quit']:
                break

            # Format causal prompt
            prompt_str = f"EN: {english_input}\nHI: "
            prompt_bytes = list(prompt_str.encode('utf-8', errors='ignore'))
            prompt_tensor = torch.tensor([prompt_bytes], dtype=torch.long, device=device)

            # Generate bytes autoregressively
            print("Translating...", end="", flush=True)
            
            with torch.no_grad():
                generated = prompt_tensor
                logits, states = model.forward(prompt_tensor)
                
                # Sample next byte
                next_logit = logits[:, -1, :] / 0.7 # temp = 0.7
                next_probs = F.softmax(next_logit, dim=-1)
                next_byte = torch.multinomial(next_probs, num_samples=1)
                
                generated = torch.cat([generated, next_byte], dim=1)
                
                offset = prompt_tensor.shape[1]
                output_bytes = []
                
                for _ in range(300):  # limit translation to 300 bytes max
                    # Check if model produced EOS (newline byte 0x0a, or padding byte 0x00)
                    byte_val = next_byte.item()
                    if byte_val == 0x0a or byte_val == 0x00:
                        break
                    
                    output_bytes.append(byte_val)
                    
                    logits, next_states = model.forward(next_byte, states, offset=offset)
                    states = next_states
                    next_logit = logits[:, -1, :] / 0.7
                    next_probs = F.softmax(next_logit, dim=-1)
                    next_byte = torch.multinomial(next_probs, num_samples=1)
                    
                    generated = torch.cat([generated, next_byte], dim=1)
                    offset += 1

            # Decode the generated Hindi UTF-8 bytes
            hindi_output = bytes(output_bytes).decode('utf-8', errors='replace').strip()
            print(f"\rHindi: {hindi_output}\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[Error during generation: {e}]")

if __name__ == "__main__":
    main()
