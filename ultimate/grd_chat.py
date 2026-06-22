"""
GRD Chat — Interactive inference for the Geometric Reservoir Delta model.

Key design:
  - Each turn: prefill model with FULL conversation history as text
  - NO recurrent state carried between turns (state carrying from bad outputs
    corrupts subsequent turns; text history is more reliable)
  - Keeps last MAX_HISTORY turns in context window

Usage:
    python ultimate/grd_chat.py --model ultimate/grd_35m_instruct.pt --size 35m --instruct
    python ultimate/grd_chat.py --model ultimate/grd_35m_v1.pt --size 35m
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import argparse

from ultimate.grd import GRDConfig, GRDModel

CONFIGS = {
    "10m":  GRDConfig(d_model=256, n_layers=4,  n_heads=4,  d_k=64, d_v=64, d_ff=512),
    "35m":  GRDConfig(d_model=384, n_layers=6,  n_heads=6,  d_k=64, d_v=64, d_ff=1024),
    "120m": GRDConfig(d_model=768, n_layers=12, n_heads=12, d_k=64, d_v=64, d_ff=2048),
}

MAX_HISTORY = 3   # keep last N (user, grd) pairs in context
MAX_CTX_BYTES = 900  # trim history to fit within this byte budget


def load_model(model_path, size, device):
    cfg   = CONFIGS[size]
    model = GRDModel(cfg).to(device)
    sd    = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    params  = sum(p.numel() for p in model.parameters())
    size_mb = os.path.getsize(model_path) / 1e6
    print(f"[*] Loaded GRD {size.upper()} ({params:,} params | {size_mb:.1f} MB)")
    return model


def build_prompt(history: list, user_input: str, instruct: bool) -> str:
    """
    Build the full prompt string from conversation history + new user input.
    history: list of (user_str, grd_str) pairs
    """
    if not instruct:
        return user_input

    lines = []
    # Trim history to fit context budget
    recent = history[-MAX_HISTORY:]
    for u, g in recent:
        lines.append(f"User: {u}")
        lines.append(f"GRD: {g}")
    lines.append(f"User: {user_input}")
    lines.append("GRD: ")

    full = "\n".join(lines)

    # Hard-trim if too long (keep suffix — more recent = more important)
    bfull = full.encode("utf-8")
    if len(bfull) > MAX_CTX_BYTES:
        bfull = bfull[-MAX_CTX_BYTES:]
        # Re-align to UTF-8 boundary
        full = bfull.decode("utf-8", errors="ignore")

    return full


@torch.no_grad()
def generate(model, prompt_text: str, max_new_bytes: int,
             temp: float, device) -> str:
    """
    Stateless generation — each call starts from zero reservoir state.
    Prefills the model with prompt_text, then samples byte by byte.
    Returns the generated string (and streams it to stdout).
    """
    temp = max(temp, 1e-5)
    prompt_bytes  = prompt_text.encode("utf-8", errors="replace")
    prompt_tensor = torch.tensor(
        list(prompt_bytes), dtype=torch.long, device=device
    ).unsqueeze(0)   # [1, T]

    # Prefill: process entire prompt in one parallel pass (kernel-accelerated)
    logits, states = model(prompt_tensor)   # states = fresh per turn

    generated_bytes = []

    # Sample + print first byte
    next_byte = torch.multinomial(
        F.softmax(logits[:, -1] / temp, dim=-1), num_samples=1
    )
    b = next_byte.item()
    generated_bytes.append(b)
    try:
        sys.stdout.write(bytes([b]).decode("utf-8"))
    except UnicodeDecodeError:
        sys.stdout.write(".")
    sys.stdout.flush()

    # Autoregressive loop — one byte per step, O(1) memory
    for _ in range(max_new_bytes - 1):
        logits, states = model(next_byte, states)
        next_byte = torch.multinomial(
            F.softmax(logits[:, -1] / temp, dim=-1), num_samples=1
        )
        b = next_byte.item()
        generated_bytes.append(b)
        try:
            sys.stdout.write(bytes([b]).decode("utf-8"))
        except UnicodeDecodeError:
            sys.stdout.write(".")
        sys.stdout.flush()

    return bytes(generated_bytes).decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="GRD Interactive Chat")
    parser.add_argument("--model",    default="ultimate/grd_35m_instruct.pt")
    parser.add_argument("--size",     default="35m", choices=["10m", "35m", "120m"])
    parser.add_argument("--max_new",  type=int,   default=200)
    parser.add_argument("--temp",     type=float, default=0.6,
                        help="Temperature: 0.3=very focused, 0.7=creative")
    parser.add_argument("--instruct", action="store_true",
                        help="Use User:/GRD: instruction format")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(args.model, args.size, device)

    print()
    print("=" * 60)
    print("   GRD  —  Geometric Reservoir Delta  —  Chat")
    print("=" * 60)
    mode = "INSTRUCT" if args.instruct else "BASE (text completion)"
    print(f"Mode  : {mode}")
    print(f"Temp  : {args.temp}  |  Max bytes: {args.max_new}")
    print("Tips  : 'reset' clears history | 'quit' exits\n")

    history = []   # list of (user_str, grd_str) pairs

    while True:
        try:
            user_input = input("You  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        if user_input.lower() == "reset":
            history = []
            print("[*] Conversation history cleared.\n")
            continue

        # Build full prompt (text history + new user turn)
        prompt = build_prompt(history, user_input, args.instruct)

        print("GRD  > ", end="", flush=True)
        response = generate(model, prompt, args.max_new, args.temp, device)
        print("\n")

        # Trim response at natural stopping points for cleaner history
        stop_at = response.find("\nUser:")
        if stop_at != -1:
            response = response[:stop_at].strip()

        # Store in history for next turn
        history.append((user_input, response.strip()))
        # Cap history length
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]


if __name__ == "__main__":
    main()
