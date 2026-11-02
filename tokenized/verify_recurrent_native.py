import os
import sys
import time
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenized.model_hgdm import TokenizedHGDM
from tokenized.model_transformer import MatchedTransformer


def count_echo_repetitions(text: str) -> float:
    """
    Computes ratio of consecutive repeated words (e.g. 'floor floor', 'the the').
    """
    words = text.lower().split()
    if len(words) < 2:
        return 0.0
    repeats = sum(1 for i in range(len(words) - 1) if words[i] == words[i + 1])
    return repeats / (len(words) - 1)


def test_model_generation(model, tokenizer, device, prompts, name="Model", max_tokens=50):
    print("\n" + "=" * 80)
    print(f"  BENCHMARK EVALUATION: {name}")
    print("=" * 80)
    
    total_repeats = 0
    total_words = 0
    total_time = 0.0
    total_gen_tokens = 0
    
    for p in prompts:
        ids = torch.tensor([tokenizer.encode(p)], device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.generate(ids, max_new_tokens=max_tokens, temperature=0.7, top_k=40)
        t1 = time.perf_counter()
        
        gen_tokens = out.shape[1] - ids.shape[1]
        dt = t1 - t0
        total_time += dt
        total_gen_tokens += gen_tokens
        
        resp = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        echo_rate = count_echo_repetitions(resp)
        words = resp.split()
        total_words += max(1, len(words))
        total_repeats += sum(1 for i in range(len(words) - 1) if words[i].lower() == words[i + 1].lower())
        
        print(f"\nPROMPT: {repr(p)}")
        print(f"GENERATION: {resp}")
        print(f"Tokens: {gen_tokens} in {dt*1000:.1f}ms ({(gen_tokens/dt):.1f} tok/s) | Echo Rate: {echo_rate*100:.1f}%")
        print("-" * 60)
        
    overall_echo = (total_repeats / max(1, total_words)) * 100
    overall_speed = total_gen_tokens / max(1e-4, total_time)
    print(f"\n[{name} SUMMARY] Avg Speed: {overall_speed:.1f} tok/s | Immediate Echo Rate: {overall_echo:.2f}%")
    return {"speed": overall_speed, "echo_rate": overall_echo}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required for verification.")
        return

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompts = [
        "1, 2, 3, 4, 5,",
        "A, B, C, D,",
        "Here are three fruits: apple, banana, orange. The first fruit is",
        "User: Explain why the sky is blue.\nAssistant:",
        "Count numbers: 10, 20, 30, 40,",
        "The quick brown fox",
    ]

    rec_native_path = args.ckpt or (
        "checkpoints_tokenized/recurrent_native/ckpt_hgdm.pt"
        if os.path.exists("checkpoints_tokenized/recurrent_native/ckpt_hgdm.pt")
        else "checkpoints_tokenized/test_smoke/ckpt_hgdm.pt"
    )
    old_hgdm_path = "checkpoints_tokenized/full_hbs/ckpt_hgdm_sft.pt"
    if not os.path.exists(old_hgdm_path):
        old_hgdm_path = "checkpoints_tokenized/ckpt_hgdm_sft.pt"
    transformer_path = "checkpoints_tokenized/ckpt_transformer_sft.pt"

    results = {}

    # 1. Recurrent-Native HGDM (if available)
    if os.path.exists(rec_native_path):
        print(f"\nLoading Recurrent-Native HGDM from {rec_native_path}...")
        d = torch.load(rec_native_path, map_location=device, weights_only=False)
        m = TokenizedHGDM(d["config"]).to(device)
        m.load_state_dict(d["model_state"], strict=False)
        m.eval()
        results["recurrent_native"] = test_model_generation(
            m, tok, device, prompts, name="HGDM Recurrent-Native (8-Band + Lateral Inh + MTP)"
        )
    else:
        print(f"Notice: {rec_native_path} not found yet (training in progress).")

    # 2. Previous HGDM-HBS SFT
    if os.path.exists(old_hgdm_path):
        print(f"\nLoading Previous HGDM-HBS from {old_hgdm_path}...")
        d = torch.load(old_hgdm_path, map_location=device, weights_only=False)
        m = TokenizedHGDM(d["config"]).to(device)
        m.load_state_dict(d["model_state"], strict=False)
        m.eval()
        results["previous_hgdm"] = test_model_generation(
            m, tok, device, prompts, name="Previous HGDM-HBS (4-Band Stateless Teacher-Forcing)"
        )

    # 3. Transformer Baseline
    if os.path.exists(transformer_path):
        print(f"\nLoading Matched Transformer from {transformer_path}...")
        d = torch.load(transformer_path, map_location=device, weights_only=False)
        m = MatchedTransformer(d["config"]).to(device)
        m.load_state_dict(d["model_state"], strict=False)
        m.eval()
        results["transformer"] = test_model_generation(
            m, tok, device, prompts, name="Transformer Baseline (KV-Cache)"
        )

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
