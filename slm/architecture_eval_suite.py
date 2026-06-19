import torch
import torch.nn.functional as F
import math
import os
import sys
import argparse
import random
from datasets import load_dataset
from tqdm import tqdm
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hgdm_omega import OmegaGDM, OmegaConfig

# Helper to calculate BPB on a subset of Wikitext
def calculate_bpb(model, data_bytes, device, chunk_size=2048, num_chunks=50):
    model.eval()
    total_loss = 0.0
    total_bytes = 0
    with torch.no_grad():
        for i in range(num_chunks):
            if (i+1)*chunk_size + 1 > len(data_bytes):
                break
            chunk = data_bytes[i*chunk_size : (i+1)*chunk_size + 1]
            x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, model.config.vocab_size), y.view(-1), reduction='sum')
            total_loss += loss.item()
            total_bytes += chunk_size
    avg_loss = total_loss / total_bytes
    bpb = avg_loss / math.log(2)
    return bpb

def run_ablation_study(model, data_bytes, device):
    print("\n=========================================")
    print("1. ARCHITECTURAL ABLATION STUDY")
    print("=========================================")
    
    # Baseline
    print("[Testing Baseline Architecture]")
    baseline_bpb = calculate_bpb(model, data_bytes, device)
    print(f"Baseline BPB: {baseline_bpb:.4f}")
    
    # Ablation 1: Zero Memory (Alpha = 0)
    print("\n[Ablation 1: Zero Memory in Semantic Core (Alpha=0)]")
    # Instead of weight hacking, we monkey patch the MultiHeadGatedDelta forward
    from ultimate.hgdm_ultimate import MultiHeadGatedDelta
    orig_forward = MultiHeadGatedDelta.forward
    
    def zero_alpha_forward(self, x, state=None):
        out, state = orig_forward(self, x, state)
        # We can't easily intercept alpha inside the compiled forward, but we can 
        # override W_alpha to output a highly negative constant cleanly
        return out, state
        
    # Let's override W_alpha cleanly
    orig_state = copy.deepcopy(model.state_dict())
    
    for module in model.modules():
        if isinstance(module, MultiHeadGatedDelta):
            # If W_alpha exists, bias=-100, weight=0 -> outputs -100 -> sigmoid(-100) = 0
            if hasattr(module, 'W_alpha'):
                module.W_alpha.weight.data.zero_()
                module.W_alpha.bias.data.fill_(-100.0)
            if hasattr(module, 'W_delta'): # Just in case
                module.W_delta.weight.data.zero_()
                module.W_delta.bias.data.fill_(100.0) # huge delta_t -> alpha = 0
                
    zero_mem_bpb = calculate_bpb(model, data_bytes, device)
    print(f"Zero Memory BPB: {zero_mem_bpb:.4f} (Degradation: +{zero_mem_bpb - baseline_bpb:.4f} BPB)")
    
    # Restore
    model.load_state_dict(orig_state)
    
    # Ablation 2: No Writing (Beta = 0)
    print("\n[Ablation 2: No Writing to Semantic Core (Beta=0)]")
    for module in model.modules():
        if isinstance(module, MultiHeadGatedDelta):
            if hasattr(module, 'W_beta'):
                module.W_beta.weight.data.zero_()
                module.W_beta.bias.data.fill_(-100.0)
                
    no_write_bpb = calculate_bpb(model, data_bytes, device)
    print(f"No Write BPB: {no_write_bpb:.4f} (Degradation: +{no_write_bpb - baseline_bpb:.4f} BPB)")
    
    # Restore
    model.load_state_dict(orig_state)

def run_needle_test(model, device):
    print("\n=========================================")
    print("2. NEEDLE-IN-A-HAYSTACK (LONG CONTEXT)")
    print("=========================================")
    
    haystack_base = (
        "The universe is vast and filled with mysteries. Stars explode and form new galaxies. "
        "Artificial Intelligence has evolved from rule-based systems to deep neural networks. "
        "Scientists recently discovered a new species of deep-sea jellyfish. "
        "The history of Rome spans thousands of years of conquest and culture. "
    ) * 8
    
    needle = " [CRITICAL ALERT: The secret passcode is: OMEGA-123] "
    
    positions = [0.1, 0.5, 0.9] # 10%, 50%, 90% depth
    
    for depth in positions:
        # Build prompt using ChatML format
        haystack_bytes = list(haystack_base.encode('utf-8'))
        insert_idx = int(len(haystack_bytes) * depth)
        
        content = bytes(haystack_bytes[:insert_idx]).decode('utf-8') + needle + bytes(haystack_bytes[insert_idx:]).decode('utf-8')
        
        prompt = (
            f"<|im_start|>user\n"
            f"Read the following text and answer the question at the end.\n\n"
            f"{content}\n\n"
            f"What is the secret passcode?<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"The secret passcode is:"
        )
        
        prompt_bytes = list(prompt.encode('utf-8'))
        x = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
        
        with torch.no_grad():
            generated = model.generate(x, max_new_bytes=10, temp=0.1)
            gen_text = bytes(generated[0, len(prompt_bytes):].cpu().tolist()).decode('utf-8', errors='replace')
        
        print(f"Depth {depth*100:.0f}% -> Answer generated: {gen_text!r}")

def run_induction_test(model, device):
    print("\n=========================================")
    print("3. IN-CONTEXT INDUCTION & FEW-SHOT")
    print("=========================================")
    
    prompt = (
        "<|im_start|>user\n"
        "Pattern matching test. Complete the final entry exactly as shown previously.\n"
        "ID: 901 -> Code: X-Alpha\n"
        "ID: 342 -> Code: Y-Beta\n"
        "ID: 777 -> Code: Z-Gamma\n"
        "ID: 342 -> Code:<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Y-Beta"
    )
    # Wait, the assistant block already has the answer if I put Y-Beta.
    # Let me fix the prompt so the assistant predicts it.
    prompt = (
        "<|im_start|>user\n"
        "Pattern matching test. Complete the final entry.\n"
        "ID: 901 -> Code: X-Alpha\n"
        "ID: 342 -> Code: Y-Beta\n"
        "ID: 777 -> Code: Z-Gamma\n"
        "ID: 342 -> Code:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    prompt_bytes = list(prompt.encode('utf-8'))
    x = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
    
    with torch.no_grad():
        generated = model.generate(x, max_new_bytes=7, temp=0.1)
        gen_text = bytes(generated[0, len(prompt_bytes):].cpu().tolist()).decode('utf-8', errors='replace')
        
    print(f"Prompt asked for 'ID: 342'. Model output: {gen_text!r}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="omega_v1_dpo_latest.pt")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cfg = OmegaConfig(
        d_byte=256, catcher_layers=2, renderer_layers=2,
        d_model=768, core_layers=12, n_heads=12,
        d_k=64, d_v=64, d_ff=3072,
        decimation_rate=8, max_position_embeddings=2048,
        vocab_size=256, use_state_fusion=False
    )
    
    print(f"Initializing OmegaGDM on {device}...")
    model = OmegaGDM(cfg, force_sequential=False).to(device)
    
    if not os.path.exists(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}")
        sys.exit(1)
        
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    
    print("Loading test data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(dataset['text'])
    data_bytes = list(text.encode('utf-8', errors='replace'))
    
    # Run Tests
    run_ablation_study(model, data_bytes, device)
    run_needle_test(model, device)
    run_induction_test(model, device)
    print("\n[Architecture Evaluation Suite Complete]")

if __name__ == "__main__":
    main()
