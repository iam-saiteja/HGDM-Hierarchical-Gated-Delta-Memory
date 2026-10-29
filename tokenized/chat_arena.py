import os
import sys
import time
import torch
import torch.nn.functional as F
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenized.config import TokenizedConfig
from tokenized.model_hgdm import TokenizedHGDM
from tokenized.model_transformer import MatchedTransformer
from transformers import AutoTokenizer


def load_model(ckpt_path: str, model_type: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        print(f"[Warning] Checkpoint not found at '{ckpt_path}'. Initializing random weights for architectural demo.")
        config = TokenizedConfig.get_tier("pilot", vocab_size=50257)
        if model_type == "hgdm":
            model = TokenizedHGDM(config)
        else:
            model = MatchedTransformer(config)
        return model.to(device), config
        
    data = torch.load(ckpt_path, map_location=device)
    config = data["config"]
    if model_type == "hgdm":
        model = TokenizedHGDM(config)
    else:
        model = MatchedTransformer(config)
    model.load_state_dict(data["model_state"])
    return model.to(device), config


def generate_with_metrics(model, model_type: str, prompt_ids: torch.Tensor, max_new_tokens: int = 128, temperature: float = 0.7):
    device = prompt_ids.device
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    t0 = time.time()
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            output_ids = model.generate(
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            
    t1 = time.time()
    elapsed = max(1e-4, t1 - t0)
    new_tokens = output_ids.shape[1] - prompt_ids.shape[1]
    speed = new_tokens / elapsed
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    
    return output_ids, speed, peak_vram


def main():
    parser = argparse.ArgumentParser(description="Conversational Arena: HGDM vs. Transformer")
    parser.add_argument("--hgdm_ckpt", type=str, default="checkpoints_tokenized/ckpt_hgdm.pt")
    parser.add_argument("--transformer_ckpt", type=str, default="checkpoints_tokenized/ckpt_transformer.pt")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--prompt", type=str, default=None, help="One-shot prompt to evaluate")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("\n[Loading Models]...")
    hgdm, cfg_h = load_model(args.hgdm_ckpt, "hgdm", device)
    trans, cfg_t = load_model(args.transformer_ckpt, "transformer", device)
    hgdm.eval()
    trans.eval()
    
    print("\n" + "=" * 75)
    print("      NATURAL-LANGUAGE CONVERSATIONAL ARENA: HGDM vs. TRANSFORMER")
    print("=" * 75)
    print("Compare responses, throughput, and memory scaling in real time.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        if args.prompt:
            user_input = args.prompt
        else:
            try:
                user_input = input("You > ").strip()
            except (KeyboardInterrupt, EOFError):
                break
                
        if not user_input or user_input.lower() in ["exit", "quit"]:
            break
            
        formatted_prompt = f"User: {user_input}\nAssistant:"
        prompt_ids = torch.tensor([tokenizer.encode(formatted_prompt)], device=device)
        
        # 1. HGDM Generation
        out_h, speed_h, vram_h = generate_with_metrics(
            hgdm, "hgdm", prompt_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature
        )
        resp_h = tokenizer.decode(out_h[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        
        # 2. Transformer Generation
        out_t, speed_t, vram_t = generate_with_metrics(
            trans, "transformer", prompt_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature
        )
        resp_t = tokenizer.decode(out_t[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        
        print("\n" + "-" * 75)
        print(f"🤖 [HGDM-Tokenized] (Recurrent Matrix State | O(1) Memory | Zero KV-Cache)")
        print(f"{resp_h.strip()}")
        print(f"\n📊 Stats: {speed_h:5.1f} tok/s | Peak VRAM: {vram_h:5.0f} MB | KV-Cache: 0 MB (Constant)")
        print("-" * 75)
        
        print(f"🧠 [Matched Transformer] (Full Self-Attention | Growing KV-Cache)")
        print(f"{resp_t.strip()}")
        print(f"\n📊 Stats: {speed_t:5.1f} tok/s | Peak VRAM: {vram_t:5.0f} MB | KV-Cache: Growing O(T)")
        print("-" * 75 + "\n")
        
        if args.prompt:
            break


if __name__ == "__main__":
    main()
