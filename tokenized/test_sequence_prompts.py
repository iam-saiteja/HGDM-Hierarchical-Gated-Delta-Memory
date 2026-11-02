import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenized.model_hgdm import TokenizedHGDM
from tokenized.model_transformer import MatchedTransformer


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load HGDM-HBS
    dh = torch.load("checkpoints_tokenized/full_hbs/ckpt_hgdm_sft.pt", map_location=device, weights_only=False)
    mh = TokenizedHGDM(dh["config"]).to(device)
    mh.load_state_dict(dh["model_state"])
    mh.eval()

    # Load Transformer
    dt = torch.load("checkpoints_tokenized/ckpt_transformer_sft.pt", map_location=device, weights_only=False)
    mt = MatchedTransformer(dt["config"]).to(device)
    mt.load_state_dict(dt["model_state"])
    mt.eval()

    test_prompts = [
        "1, 2, 3, 4, 5,",
        "Count from 1 to 10: 1, 2, 3, 4,",
        "A, B, C, D,",
        "Monday, Tuesday, Wednesday,",
        "Complete the pattern: 2, 4, 6, 8,",
        "The cat sat on the",
        "Question: What is 2 + 2? Answer:",
        "User: Hi! How are you?\nAssistant:",
        "Here are three fruits: apple, banana, orange. The first fruit is"
    ]

    print("=" * 80)
    print("      STRUCTURED & SEQUENTIAL COMPLETION: HGDM-HBS vs. TRANSFORMER")
    print("=" * 80)

    for p in test_prompts:
        ids = torch.tensor([tok.encode(p)], device=device)
        print("\n" + "-" * 80)
        print(f"PROMPT: {repr(p)}")
        print("-" * 80)

        # Transformer sampling (temp=0.7)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_t_sample = mt.generate(ids, max_new_tokens=50, temperature=0.7)
                resp_t_sample = tok.decode(out_t_sample[0, ids.shape[1]:], skip_special_tokens=True).strip()

        # Transformer greedy (temp=0.1)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_t_greedy = mt.generate(ids, max_new_tokens=50, temperature=0.1)
                resp_t_greedy = tok.decode(out_t_greedy[0, ids.shape[1]:], skip_special_tokens=True).strip()

        # HGDM sampling (temp=0.7)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_h_sample = mh.generate(ids, max_new_tokens=50, temperature=0.7)
                resp_h_sample = tok.decode(out_h_sample[0, ids.shape[1]:], skip_special_tokens=True).strip()

        # HGDM greedy (temp=0.1)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_h_greedy = mh.generate(ids, max_new_tokens=50, temperature=0.1)
                resp_h_greedy = tok.decode(out_h_greedy[0, ids.shape[1]:], skip_special_tokens=True).strip()

        print(f"[Transformer  - Sampling T=0.7]:\n{resp_t_sample}\n")
        print(f"[Transformer  - Greedy   T=0.1]:\n{resp_t_greedy}\n")
        print(f"[HGDM-HBS     - Sampling T=0.7]:\n{resp_h_sample}\n")
        print(f"[HGDM-HBS     - Greedy   T=0.1]:\n{resp_h_greedy}\n")


if __name__ == "__main__":
    main()
