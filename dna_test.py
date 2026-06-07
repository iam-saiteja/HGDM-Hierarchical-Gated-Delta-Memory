import torch
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hgdm_omega import OmegaGDM, OmegaConfig

def main():
    parser = argparse.ArgumentParser(description="Test HGDM DNA Sequence Generator")
    parser.add_argument("--ckpt", default="hgdm_dna_latest.pt", help="Path to checkpoint .pt file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Running device (cuda or cpu)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[System] Using device: {device}")

    if not os.path.exists(args.ckpt):
        print(f"[Error] Checkpoint file '{args.ckpt}' not found. Please train the model first!")
        sys.exit(1)

    print(f"[System] Loading checkpoint from '{args.ckpt}'...")
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    
    if 'config' in checkpoint:
        cfg = checkpoint['config']
    else:
        print("[System] Config not found in checkpoint. Using default DNA configuration...")
        cfg = OmegaConfig(
            d_byte=256, catcher_layers=1, renderer_layers=1,
            d_model=512, core_layers=8, n_heads=8,
            d_k=64, d_v=64, d_ff=2048,
            decimation_rate=8, max_position_embeddings=8192,
            vocab_size=256, use_state_fusion=False
        )

    model = OmegaGDM(cfg, force_sequential=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    step = checkpoint.get('step', '?')
    tokens = checkpoint.get('tokens_trained', '?')
    print(f"[System] Model Loaded successfully! Trained on {tokens:,} bases (Step {step})")
    print(f"[System] Ready! Type 'exit' or 'quit' to close.")
    print("─" * 70)

    while True:
        try:
            dna_input = input("\nEnter seed DNA sequence (A, C, G, T): ").strip()
            if not dna_input:
                continue
            if dna_input.lower() in ['exit', 'quit']:
                break
                
            # Filter seed to standard uppercase DNA characters
            dna_clean = "".join([c.upper() for c in dna_input if c.upper() in ('A', 'C', 'G', 'T', 'N')])
            if not dna_clean:
                print("[Warning] No valid DNA bases detected in input! Please use A, C, G, T, or N.")
                continue
                
            print(f"Validated Seed DNA  : {dna_clean}")
            prompt_bytes = list(dna_clean.encode('ascii'))
            prompt_tensor = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
            
            # Autoregressive generation of 200 bases
            with torch.no_grad():
                generated = model.generate(prompt_tensor, max_new_bytes=200, temp=0.5)
                
            gen_bytes = generated[0, len(prompt_bytes):].cpu().tolist()
            
            # Decode to string, keeping only DNA characters
            dna_chars = []
            for x in gen_bytes:
                char = chr(x).upper()
                if char in ('A', 'C', 'G', 'T', 'N'):
                    dna_chars.append(char)
                else:
                    dna_chars.append('.')
                    
            decoded_dna = "".join(dna_chars)
            print(f"Generated Continuation: {decoded_dna}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"[Generation Error] {e}")

if __name__ == "__main__":
    main()
