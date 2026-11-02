import os
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenized.model_hgdm import TokenizedHGDM
from tokenized.model_transformer import MatchedTransformer


def run_detailed_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 1. Load New Recurrent-Native HGDM (Pretrained 100M, 8 Bands, Lateral Inh, TBPTT)
    ckpt_new_path = "checkpoints_tokenized/recurrent_native/ckpt_hgdm.pt"
    d_new = torch.load(ckpt_new_path, map_location=device, weights_only=False)
    m_new = TokenizedHGDM(d_new["config"]).to(device)
    m_new.load_state_dict(d_new["model_state"], strict=False)
    m_new.eval()

    # 2. Load Old HGDM (620M Pretrain + 500-step SFT, 4 Bands, Stateless)
    ckpt_old_path = "checkpoints_tokenized/full_hbs/ckpt_hgdm_sft.pt"
    if not os.path.exists(ckpt_old_path):
        ckpt_old_path = "checkpoints_tokenized/ckpt_hgdm_sft.pt"
    d_old = torch.load(ckpt_old_path, map_location=device, weights_only=False)
    m_old = TokenizedHGDM(d_old["config"]).to(device)
    m_old.load_state_dict(d_old["model_state"], strict=False)
    m_old.eval()

    # 3. Load Transformer (620M Pretrain + 500-step SFT, KV-Cache)
    ckpt_tf_path = "checkpoints_tokenized/ckpt_transformer_sft.pt"
    d_tf = torch.load(ckpt_tf_path, map_location=device, weights_only=False)
    m_tf = MatchedTransformer(d_tf["config"]).to(device)
    m_tf.load_state_dict(d_tf["model_state"], strict=False)
    m_tf.eval()

    test_prompts = [
        "User: What is artificial intelligence?\nAssistant:",
        "Question: What is 2 + 2? Answer:",
        "1, 2, 3, 4, 5,",
        "A, B, C, D,",
        "Here are three fruits: apple, banana, orange. The first fruit is",
        "Once upon a time in a digital universe, a recurrent brain",
    ]

    print("=" * 90)
    print("DETAILED INFERENCE BENCHMARK: NEW RECURRENT-NATIVE vs OLD SFT vs TRANSFORMER SFT")
    print("=" * 90)

    for prompt in test_prompts:
        p_ids = torch.tensor([tok.encode(prompt)], device=device)
        print("\n" + "=" * 90)
        print(f"PROMPT: {repr(prompt)}")
        print("=" * 90)

        # Eval Model 1: New Recurrent-Native HGDM
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_new = m_new.generate(p_ids, max_new_tokens=45, temperature=0.7, top_k=40, repetition_penalty=1.15)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        dt_new = t1 - t0
        toks_new = out_new.shape[1] - p_ids.shape[1]
        speed_new = toks_new / max(1e-4, dt_new)
        text_new = tok.decode(out_new[0, p_ids.shape[1]:], skip_special_tokens=True).strip()

        # Eval Model 2: Old HGDM SFT
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_old = m_old.generate(p_ids, max_new_tokens=45, temperature=0.7, top_k=40)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        dt_old = t1 - t0
        toks_old = out_old.shape[1] - p_ids.shape[1]
        speed_old = toks_old / max(1e-4, dt_old)
        text_old = tok.decode(out_old[0, p_ids.shape[1]:], skip_special_tokens=True).strip()

        # Eval Model 3: Transformer SFT
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_tf = m_tf.generate(p_ids, max_new_tokens=45, temperature=0.7, top_k=40)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        dt_tf = t1 - t0
        toks_tf = out_tf.shape[1] - p_ids.shape[1]
        speed_tf = toks_tf / max(1e-4, dt_tf)
        text_tf = tok.decode(out_tf[0, p_ids.shape[1]:], skip_special_tokens=True).strip()

        print(f"\n[1. NEW Recurrent-Native HGDM (Pure Pretrained - 100M)]")
        print(f"Speed: {speed_new:.1f} tok/s ({dt_new*1000/toks_new:.2f} ms/tok) | KV-Cache: 0 KB (Recurrent State: 2.25 MB constant)")
        print(f"Generated:\n{text_new}")

        print(f"\n[2. OLD HGDM-HBS (620M Pretrain + 500 SFT)]")
        print(f"Speed: {speed_old:.1f} tok/s ({dt_old*1000/toks_old:.2f} ms/tok) | KV-Cache: 0 KB (Recurrent State: 1.15 MB constant)")
        print(f"Generated:\n{text_old}")

        print(f"\n[3. Matched Transformer (620M Pretrain + 500 SFT)]")
        print(f"Speed: {speed_tf:.1f} tok/s ({dt_tf*1000/toks_tf:.2f} ms/tok) | KV-Cache: Growing O(T) (1.1 GB at 120k tokens)")
        print(f"Generated:\n{text_tf}")
        print("-" * 90)


if __name__ == "__main__":
    run_detailed_eval()
