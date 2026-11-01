import time
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenized.config import TokenizedConfig
from tokenized.model_hgdm import TokenizedHGDM
from transformers import AutoTokenizer

def profile_breakdown():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")
    ckpt_path = "checkpoints_tokenized/ckpt_hgdm_sft.pt"
    data = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = data["config"]
    model = TokenizedHGDM(cfg).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "User: Explain how the brain processes language.\nAssistant:"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)

    # Prime states
    _, states = model(prompt_ids)
    curr_token = prompt_ids[:, -1:]

    # Measure components over 500 steps
    N = 500
    
    # 1. Mixer alone (linear proj + fused_nitro_step)
    layer0 = model.layers[0]
    x_emb = model.embedding(curr_token)
    x_norm = layer0.norm1(x_emb)
    state0 = states[0]

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        _, state0 = layer0.mixer(x_norm, state=state0)
    torch.cuda.synchronize()
    t_mixer_1layer = (time.time() - t0) / N

    # 2. SwiGLU alone
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        _ = layer0.mlp(x_norm)
    torch.cuda.synchronize()
    t_mlp_1layer = (time.time() - t0) / N

    # 3. LM Head projection (384 -> 50,257)
    x_final = model.norm_f(x_emb)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        logits = model.lm_head(x_final)
    torch.cuda.synchronize()
    t_lm_head = (time.time() - t0) / N

    # 4. Sampling: top_k=50 + multinomial vs argmax
    logits_step = logits[:, -1, :] / 0.7
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        v, _ = torch.topk(logits_step, 50)
        mask = logits_step < v[:, [-1]]
        probs = torch.softmax(logits_step.masked_fill(mask, -float('inf')), dim=-1)
        _ = torch.multinomial(probs, 1)
    torch.cuda.synchronize()
    t_multinomial = (time.time() - t0) / N

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        _ = torch.argmax(logits_step, dim=-1, keepdim=True)
    torch.cuda.synchronize()
    t_argmax = (time.time() - t0) / N

    # 5. Full step forward
    torch.cuda.synchronize()
    t0 = time.time()
    s_cur = states
    for _ in range(N):
        logits, s_cur = model(curr_token, states=s_cur)
    torch.cuda.synchronize()
    t_full_model_fwd = (time.time() - t0) / N

    print("\n" + "=" * 70)
    print("      PER-TOKEN INFERENCE LATENCY BREAKDOWN (RTX 3090 Ti)")
    print("=" * 70)
    print(f"Mixer (1 Layer):         {t_mixer_1layer*1000:6.3f} ms  ({1/t_mixer_1layer:8.0f} ops/s)")
    print(f"All 6 Mixers Total:      {t_mixer_1layer*6*1000:6.3f} ms")
    print(f"All 6 SwiGLU MLPs Total: {t_mlp_1layer*6*1000:6.3f} ms")
    print(f"LM Head (50k Vocab):     {t_lm_head*1000:6.3f} ms")
    print(f"Full Model Forward:      {t_full_model_fwd*1000:6.3f} ms  ({1/t_full_model_fwd:8.1f} tok/s pure model)")
    print("-" * 70)
    print(f"Sampling (top_k + multi):{t_multinomial*1000:6.3f} ms  (Adds {t_multinomial*1000:5.2f} ms per token!)")
    print(f"Sampling (Greedy argmax):{t_argmax*1000:6.3f} ms")
    print("=" * 70)
    print(f"Throughput with Multinomial: {1 / (t_full_model_fwd + t_multinomial):.1f} tok/s")
    print(f"Throughput with Argmax:      {1 / (t_full_model_fwd + t_argmax):.1f} tok/s")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    profile_breakdown()
