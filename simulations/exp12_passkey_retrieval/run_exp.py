import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hgdm_ultimate import HGDMUltimate, HGDMConfig

def generate_passkey_batch(batch_size, seq_len, device):
    """
    Generates a batch of noise with a hidden passkey.
    Format: [noise] The passkey is X. [noise] The passkey is X
    Where X is a single digit 0-9.
    """
    # Create random lowercase letters as noise
    x = torch.randint(97, 122, (batch_size, seq_len), device=device, dtype=torch.long)
    y = x.clone() # We only care about the final prediction, but we need y for cross entropy
    
    for b in range(batch_size):
        passkey = str(random.randint(0, 9))
        passkey_byte = ord(passkey)
        
        # Insert the passkey randomly in the first half of the sequence
        prompt_1 = "The passkey is " + passkey + ". "
        p1_bytes = [ord(c) for c in prompt_1]
        
        max_idx = (seq_len // 2) - len(p1_bytes)
        if max_idx < 0: max_idx = 0
        insert_idx = random.randint(0, max_idx)
        
        x[b, insert_idx:insert_idx+len(p1_bytes)] = torch.tensor(p1_bytes, device=device)
        
        # Insert the question at the very end
        prompt_2 = "What is the passkey? "
        p2_bytes = [ord(c) for c in prompt_2]
        
        # The sequence ends with the prompt, the target y ends with the passkey
        end_idx = seq_len - 1
        start_q = end_idx - len(p2_bytes)
        
        x[b, start_q:end_idx] = torch.tensor(p2_bytes, device=device)
        
        # The target at the very last position should be the passkey byte
        y[b, end_idx] = passkey_byte
        
        # To avoid penalizing noise predictions, we can use a loss mask later,
        # but for simplicity we will just compute CE on the last token.
        
    return x, y

def train_curriculum(model, opt, device):
    print("="*60)
    print("EXP 12: PASSKEY RETRIEVAL (THE NEEDLE TEST)")
    print("="*60)
    
    # Phase 1: 256
    # Phase 2: 1024
    # Phase 3: 4096
    curriculum = [
        {"len": 256, "steps": 500},
        {"len": 1024, "steps": 500},
        {"len": 4096, "steps": 500}
    ]
    
    model.train()
    scaler = torch.amp.GradScaler('cuda')
    
    total_step = 0
    for phase in curriculum:
        seq_len = phase["len"]
        steps = phase["steps"]
        print(f"\n--- Phase Context Length: {seq_len} ---")
        
        t0 = time.time()
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            
            x, y = generate_passkey_batch(4 if seq_len <= 1024 else 1, seq_len, device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)[0]
                # We only calculate loss on the final token prediction
                last_logits = logits[:, -1, :] # (B, vocab_size)
                last_targets = y[:, -1]        # (B,)
                loss = F.cross_entropy(last_logits, last_targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            
            if step % 100 == 0:
                pred = last_logits.argmax(dim=-1)
                acc = (pred == last_targets).float().mean().item() * 100
                print(f"Step {step:4d} | Loss: {loss.item():.4f} | Accuracy: {acc:5.1f}%")
                
        print(f"Phase Complete in {time.time()-t0:.1f}s")
        
def evaluate_grid(model, device):
    print("\n--- EVALUATION GRID ---")
    model.eval()
    
    lengths = [1024, 2048, 4096, 8192]
    depths = [0.1, 0.5, 0.9] # How far into the context the passkey is hidden
    
    results_grid = {}
    
    with torch.no_grad():
        for L in lengths:
            results_grid[L] = {}
            for D in depths:
                correct = 0
                total = 10
                
                for _ in range(total):
                    x = torch.randint(97, 122, (1, L), device=device, dtype=torch.long)
                    passkey = str(random.randint(0, 9))
                    
                    p1 = "The passkey is " + passkey + ". "
                    insert_idx = int(L * D)
                    # ensure it fits
                    if insert_idx + len(p1) > L - 50:
                        insert_idx = L - 50 - len(p1)
                        
                    x[0, insert_idx:insert_idx+len(p1)] = torch.tensor([ord(c) for c in p1], device=device)
                    
                    p2 = "What is the passkey? "
                    start_q = L - len(p2) - 1
                    x[0, start_q:L-1] = torch.tensor([ord(c) for c in p2], device=device)
                    
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        logits = model(x)[0]
                    
                    pred_byte = logits[0, -2, :].argmax().item()
                    if chr(pred_byte) == passkey:
                        correct += 1
                        
                acc = (correct / total) * 100
                results_grid[L][D] = acc
                print(f"L={L:4d} | Depth={D:.1f} | Accuracy: {acc:5.1f}%")
                
    return results_grid

def run_experiment():
    device = torch.device('cuda')
    config = HGDMConfig(
        d_model=768,
        n_layers=6, # Smaller model for rapid testing
        n_heads=12,
        vocab_size=256
    )
    model = HGDMUltimate(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4)
    
    train_curriculum(model, opt, device)
    grid = evaluate_grid(model, device)
    
    os.makedirs("results", exist_ok=True)
    with open("results/results.json", "w") as f:
        json.dump(grid, f, indent=4)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required.")
        sys.exit(1)
    run_experiment()
