"""
GRD Chat — Interactive inference for the Geometric Reservoir Delta model.
Usage:
    python ultimate/grd_chat.py
    python ultimate/grd_chat.py --model ultimate/grd_35m_v1.pt --size 35m
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import argparse

from ultimate.grd import GRDConfig, GRDModel

# ── Size presets (must match what was used during training) ───────────────────
CONFIGS = {
    "10m": GRDConfig(d_model=256, n_layers=4, n_heads=4, d_k=64, d_v=64, d_ff=512),
    "35m": GRDConfig(d_model=384, n_layers=6, n_heads=6, d_k=64, d_v=64, d_ff=1024),
    "120m": GRDConfig(d_model=768, n_layers=12, n_heads=12, d_k=64, d_v=64, d_ff=2048),
}

def load_model(model_path, size, device):
    cfg   = CONFIGS[size]
    model = GRDModel(cfg).to(device)
    sd    = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    size_mb = os.path.getsize(model_path) / 1e6
    print(f"[*] Loaded GRD {size.upper()} ({params:,} params | {size_mb:.1f} MB)")
    return model

@torch.no_grad()
def generate(model, prompt_text: str, max_new_bytes: int, temp: float, device):
    """Run auto-regressive byte-level generation."""
    prompt_bytes = prompt_text.encode("utf-8", errors="replace")
    prompt_tensor = torch.tensor(
        list(prompt_bytes), dtype=torch.long, device=device
    ).unsqueeze(0)  # [1, T]

    # Prefill: encode the prompt and get the recurrent state
    logits, states = model(prompt_tensor)

    generated = list(prompt_bytes)
    # Sample first new byte from end of prompt
    next_byte = torch.multinomial(
        F.softmax(logits[:, -1] / temp, dim=-1), num_samples=1
    )
    generated.append(next_byte.item())

    # Auto-regressive generation: one byte at a time, O(1) per step
    for _ in range(max_new_bytes - 1):
        logits, states = model(next_byte, states)
        next_byte = torch.multinomial(
            F.softmax(logits[:, -1] / temp, dim=-1), num_samples=1
        )
        b = next_byte.item()
        generated.append(b)
        # Print as we go for streaming feel
        try:
            sys.stdout.write(bytes([b]).decode("utf-8"))
            sys.stdout.flush()
        except UnicodeDecodeError:
            sys.stdout.write(".")
            sys.stdout.flush()

    return bytes(generated[len(prompt_bytes):]).decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="GRD Interactive Chat")
    parser.add_argument("--model", default="ultimate/grd_35m_v1.pt",
                        help="Path to the .pt checkpoint")
    parser.add_argument("--size",    default="35m", choices=["10m", "35m", "120m"])
    parser.add_argument("--max_new", type=int,   default=200,
                        help="Max bytes to generate per response")
    parser.add_argument("--temp",    type=float, default=0.8,
                        help="Sampling temperature (lower = less random)")
    parser.add_argument("--instruct", action="store_true",
                        help="Wrap prompts in 'User:/GRD:' format (use with instruct checkpoints)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.model, args.size, device)

    print()
    print("=" * 60)
    print("   GRD  —  Geometric Reservoir Delta  —  Interactive")
    print("=" * 60)
    if args.instruct:
        print("Mode: INSTRUCT  (User/GRD format, carries memory across turns)")
    else:
        print("Mode: BASE  (raw text continuation)")
    print("Type a prompt and press Enter. Type 'quit' to exit.\n")

    conversation_states = None   # carry recurrent state across turns

    while True:
        try:
            prompt = input("You  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        if prompt.lower() == "reset":
            conversation_states = None
            print("[*] Memory reset.\n")
            continue

        # Format the prompt correctly for instruct models
        if args.instruct:
            # Wrap in the same template used during finetune_grd.py training
            formatted_prompt = f"User: {prompt}\nGRD: "
        else:
            formatted_prompt = prompt

        print(f"GRD  > ", end="", flush=True)
        generate(model, formatted_prompt, args.max_new, args.temp, device)
        print("\n")


if __name__ == "__main__":
    main()
