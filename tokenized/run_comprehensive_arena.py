import os
import sys
import time
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tokenized.config import TokenizedConfig
from tokenized.model_hgdm import TokenizedHGDM
from tokenized.model_transformer import MatchedTransformer

def load_models(device, ckpt_h: str, ckpt_t: str):
    data_h = torch.load(ckpt_h, map_location=device, weights_only=False)
    cfg_h = data_h["config"]
    model_h = TokenizedHGDM(cfg_h)
    model_h.load_state_dict(data_h["model_state"])
    model_h.to(device).eval()
    
    data_t = torch.load(ckpt_t, map_location=device, weights_only=False)
    cfg_t = data_t["config"]
    model_t = MatchedTransformer(cfg_t)
    model_t.load_state_dict(data_t["model_state"])
    model_t.to(device).eval()
    
    return model_h, model_t, cfg_h

def generate_tokens(model, prompt_ids, max_new_tokens=64, temperature=0.7):
    device = prompt_ids.device
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            output_ids = model.generate(
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    elapsed = max(1e-4, t1 - t0)
    new_tokens = output_ids.shape[1] - prompt_ids.shape[1]
    speed = new_tokens / elapsed
    return output_ids, elapsed, speed, new_tokens

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hgdm_ckpt", type=str, default="checkpoints_tokenized/ckpt_hgdm.pt")
    parser.add_argument("--transformer_ckpt", type=str, default="checkpoints_tokenized/ckpt_transformer.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model_h, model_t, cfg = load_models(device, args.hgdm_ckpt, args.transformer_ckpt)
    
    # State sizes
    hgdm_state_bytes = cfg.n_layers * cfg.n_heads * cfg.d_k * cfg.d_v * 2  # bfloat16
    hgdm_state_kb = hgdm_state_bytes / 1024.0
    
    prompts = [
        {
            "category": "Exact Key-Value Association (Stress Test: Linear Interference)",
            "prompt": "Here is a list of secret agent code names: Alice is Falcon, Bob is Shadow, Charlie is Viper, David is Ghost. Question: What is the secret code name of Charlie? Answer:",
            "max_tokens": 40
        },
        {
            "category": "Algorithmic / Bracket Matching (Stress Test: Stack Memory)",
            "prompt": "Complete the matching brackets in order: { [ ( a + b ) * c ] - [ d / ( e + f ) ] } * { [ (",
            "max_tokens": 30
        },
        {
            "category": "Verbatim Sequence Inversion (Stress Test: Backtracking)",
            "prompt": "Reverse the order of these words: orange, banana, apple, grape, cherry. Output:",
            "max_tokens": 35
        },
        {
            "category": "Needle in Haystack (Stress Test: State Overwriting & Decay)",
            "prompt": "The library was silent. Outside the rain fell against the high stained-glass windows. Important note: The hidden gate code is 7492. Many researchers sat reading old encyclopedias and discussing philosophy until dawn. Question: What is the hidden gate code? Answer:",
            "max_tokens": 30
        },
        {
            "category": "Factual Technical Knowledge (Fair Baseline Comparison)",
            "prompt": "Explain the difference between supervised learning and unsupervised learning in machine learning.",
            "max_tokens": 90
        },
        {
            "category": "Sequential Counting / State Progression",
            "prompt": "Count from 1 to 20 separated by commas: 1, 2, 3, 4, 5,",
            "max_tokens": 45
        }
    ]
    
    results = []
    
    for idx, p_info in enumerate(prompts):
        raw_prompt = p_info["prompt"]
        max_tokens = p_info["max_tokens"]
        category = p_info["category"]
        
        prompt_ids = torch.tensor([tokenizer.encode(raw_prompt)], device=device)
        t_prompt = prompt_ids.shape[1]
        
        # Run HGDM
        out_h, time_h, speed_h, gen_h = generate_tokens(model_h, prompt_ids, max_new_tokens=max_tokens)
        resp_h = tokenizer.decode(out_h[0, t_prompt:], skip_special_tokens=True).strip()
        
        # Run Transformer
        out_t, time_t, speed_t, gen_t = generate_tokens(model_t, prompt_ids, max_new_tokens=max_tokens)
        resp_t = tokenizer.decode(out_t[0, t_prompt:], skip_special_tokens=True).strip()
        
        t_total_h = t_prompt + gen_h
        t_total_t = t_prompt + gen_t
        
        # Transformer KV cache at total context: 2 * L * H * d_k * T_total * 2 bytes
        trans_kv_bytes = 2 * cfg.n_layers * cfg.n_heads * cfg.d_k * t_total_t * 2
        trans_kv_kb = trans_kv_bytes / 1024.0
        ratio = trans_kv_kb / hgdm_state_kb
        
        res_item = {
            "index": idx + 1,
            "category": category,
            "prompt": raw_prompt,
            "prompt_tokens": t_prompt,
            "hgdm": {
                "output": resp_h,
                "time_sec": time_h,
                "tok_per_sec": speed_h,
                "gen_tokens": gen_h,
                "state_kb": hgdm_state_kb
            },
            "transformer": {
                "output": resp_t,
                "time_sec": time_t,
                "tok_per_sec": speed_t,
                "gen_tokens": gen_t,
                "kv_cache_kb": trans_kv_kb
            },
            "savings_ratio": ratio
        }
        results.append(res_item)
        
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
