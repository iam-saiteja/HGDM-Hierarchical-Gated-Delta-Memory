import torch
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hgdm_omega import OmegaGDM, OmegaConfig

def main():
    parser = argparse.ArgumentParser(description="Test HGDM English-to-Hindi Translator")
    parser.add_argument("--ckpt", default="hgdm_translation_latest.pt", help="Path to checkpoint .pt file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Running device (cuda or cpu)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[System] Using device: {device}")

    if not os.path.exists(args.ckpt):
        print(f"[Error] Checkpoint file '{args.ckpt}' not found. Please train the model first!")
        sys.exit(1)

    print(f"[System] Loading checkpoint from '{args.ckpt}'...")
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=True)
    
    # Load configuration dynamically from checkpoint if saved, else use default 39.5M config
    if 'config' in checkpoint:
        cfg = checkpoint['config']
    else:
        print("[System] Config not found in checkpoint. Using default 39.5M configuration...")
        cfg = OmegaConfig(
            d_byte=256, catcher_layers=1, renderer_layers=1,
            d_model=512, core_layers=8, n_heads=8,
            d_k=64, d_v=64, d_ff=2048,
            decimation_rate=8, max_position_embeddings=2048,
            vocab_size=256, use_state_fusion=False
        )

    model = OmegaGDM(cfg, force_sequential=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    step = checkpoint.get('step', '?')
    tokens = checkpoint.get('tokens_trained', '?')
    print(f"[System] Model Loaded successfully! Trained on {tokens:,} target tokens (Step {step})")
    print(f"[System] Ready! Type 'exit' or 'quit' to close.")
    print("─" * 70)

    while True:
        try:
            en_input = input("\nEnter English sentence: ").strip()
            if not en_input:
                continue
            if en_input.lower() in ['exit', 'quit']:
                break
                
            # Construct causal prompt
            prompt = f"EN: {en_input}\nHI: "
            prompt_bytes = list(prompt.encode('utf-8', errors='ignore'))
            prompt_tensor = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
            
            # Autoregressive generation
            with torch.no_grad():
                # Hindi outputs are typically 2-3x longer in bytes than English due to 3-byte Devanagari UTF-8 encoding
                generated = model.generate(prompt_tensor, max_new_bytes=200, temp=0.3)
                
            gen_bytes = generated[0, len(prompt_bytes):].cpu().tolist()
            
            # Truncate at newline '\n' (0x0a) or padding byte 0x00
            if 10 in gen_bytes:
                gen_bytes = gen_bytes[:gen_bytes.index(10)]
            elif 0 in gen_bytes:
                gen_bytes = gen_bytes[:gen_bytes.index(0)]
                
            decoded_hindi = bytes(gen_bytes).decode('utf-8', errors='replace').strip()
            
            print(f"Hindi Translation  : {decoded_hindi}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"[Generation Error] {e}")

if __name__ == "__main__":
    main()
