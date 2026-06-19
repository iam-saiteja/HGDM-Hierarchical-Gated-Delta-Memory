import torch
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hgdm_omega import OmegaGDM, OmegaConfig

def decode_byte(b):
    try:
        char = bytes([b]).decode('utf-8')
        if char.isprintable() or char in ['\n', '\t']:
            return char
        else:
            return "" # Hide unprintables for a clean chat experience
    except UnicodeDecodeError:
        return ""

def main():
    parser = argparse.ArgumentParser(description="Omega Edge (Watch Edition) Chat Interface")
    parser.add_argument('--ckpt', type=str, default='omega_edge_v1.pt', help='Path to the fine-tuned edge model')
    parser.add_argument('--max_new_bytes', type=int, default=150, help='Maximum real bytes to generate')
    parser.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Loading Omega Edge (35M) from {args.ckpt}...")
    
    # 35M Parameter Configuration
    config = OmegaConfig(
        d_byte=256, catcher_layers=2, renderer_layers=2, 
        d_model=256, core_layers=6, n_heads=8, 
        d_k=32, d_v=32, d_ff=1024, decimation_rate=8, 
        max_position_embeddings=512, vocab_size=256, use_state_fusion=False
    )
    
    model = OmegaGDM(config, force_sequential=False).to(device)
    
    ckpt_path = os.path.join(os.path.dirname(__file__), args.ckpt)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print("[*] Weights loaded successfully.")
    else:
        print(f"[!] Critical Error: Checkpoint {ckpt_path} not found! Please run finetune_edge.py first.")
        return
        
    model.eval()
    model.bfloat16()

    print("\n" + "="*60)
    print("   OMEGA EDGE (Watch V1) NATIVE CHAT")
    print("="*60)
    print("Type 'quit' to exit.\n")

    # Conversation history buffer (keeps recent context for the chat)
    history = ""

    while True:
        try:
            prompt = input("\nUser: ")
            if not prompt.strip():
                continue
            if prompt.lower() in ['quit', 'exit']:
                break
        except (EOFError, KeyboardInterrupt):
            break

        # Format exactly like training data
        history += f"User: {prompt}\nOmega: "
        
        # Keep context window small to avoid OOM or slow inference
        # In a real watch, we'd truncate to the last 2000 bytes
        if len(history) > 2000:
            history = history[-2000:]
            
        history_bytes = list(history.encode('utf-8', errors='ignore'))
        
        # Add 5 "Dreaming" bytes because the model was trained to expect them!
        dream_bytes = [0x00] * 5
        full_input = history_bytes + dream_bytes
        
        x = torch.tensor([full_input], dtype=torch.long, device=device)
        
        print("Omega: ", end='', flush=True)
        
        states = None
        offset = 0
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Process full prompt + dream bytes
                logits, states = model(byte_seq=x, states=states, offset=offset)
                offset += x.shape[1]
                
                # Start generating
                next_byte = torch.argmax(logits[:, -1, :], dim=-1).item()
                char = decode_byte(next_byte)
                print(char, end='', flush=True)
                
                response_str = char
                current_byte = torch.tensor([[next_byte]], dtype=torch.long, device=device)
                
                for _ in range(args.max_new_bytes):
                    logits, states = model(byte_seq=current_byte, states=states, offset=offset)
                    
                    scaled_logits = logits[:, -1, :] / args.temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    next_byte = torch.multinomial(probs, num_samples=1).item()
                    
                    char = decode_byte(next_byte)
                    print(char, end='', flush=True)
                    response_str += char
                    
                    # Stop if we hit a double newline (typical Alpaca ending)
                    if response_str.endswith("\n\n"):
                        break
                        
                    current_byte = torch.tensor([[next_byte]], dtype=torch.long, device=device)
                    offset += 1
                    
        # Append response to history so it remembers the conversation
        history += response_str
        print()

if __name__ == "__main__":
    main()
