import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import time
import json
import random
from hgdm_ultimate import HGDMUltimate, HGDMConfig
from utils import get_gpu_memory_usage

def generate_passkey_data(batch_size, seq_len, depth, device):
    """
    Generates synthetic passkey data.
    depth: float between 0.0 and 1.0 indicating where to place the passkey.
    Returns x (inputs) and y (targets).
    """
    x_batch = []
    y_batch = []
    keys = []
    
    for _ in range(batch_size):
        passkey = f"{random.randint(10000, 99999)}"
        keys.append(passkey)
        passkey_str = f" The passkey is {passkey}. "
        prompt_str = f" What is the passkey? {passkey}"
        
        fixed_len = len(passkey_str) + len(prompt_str)
        noise_len = max(0, seq_len - fixed_len)
        
        pos = int(noise_len * depth)
        
        # Noise: random lowercase ASCII letters
        noise1 = torch.randint(97, 123, (pos,), dtype=torch.uint8)
        noise2 = torch.randint(97, 123, (noise_len - pos,), dtype=torch.uint8)
        
        p1 = torch.tensor(list(passkey_str.encode('utf-8')), dtype=torch.uint8)
        p2 = torch.tensor(list(prompt_str.encode('utf-8')), dtype=torch.uint8)
        
        seq = torch.cat([noise1, p1, noise2, p2])
        x_batch.append(seq[:-1])
        y_batch.append(seq[1:])
        
    x = torch.stack(x_batch).long().to(device)
    y = torch.stack(y_batch).long().to(device)
    
    return x, y, keys

def train_passkey_curriculum():
    device = torch.device('cuda')
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    model = HGDMUltimate(config).to(device)
    
    checkpoint_path = "../exp1_enwik8/hgdm_enwik8_120M.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading base Enwik8 checkpoint from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    else:
        print("WARNING: Checkpoint not found. Training from scratch will fail to learn retrieval.")
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n{'='*50}\nExp 11: Passkey Retrieval (Context Window Test)\n{'='*50}")
    
    # Extended curriculum with a mastery phase for deep convergence
    curriculum = [
        (512, 500),
        (1024, 500),
        (2048, 500),
        (4096, 500),
        (4096, 1000)   # mastery phase
    ]
    
    t_start = time.time()
    
    for seq_len, steps in curriculum:
        print(f"\n--- Curriculum Phase: Seq Len {seq_len} ({steps} steps) ---")
        if seq_len == 4096 and steps > 500:
            for g in opt.param_groups:
                g['lr'] = 3e-4
        for step in range(steps):
            depth = random.uniform(0.1, 0.9)
            x, y, _ = generate_passkey_data(2, seq_len, depth, device)
            
            opt.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(x)
                loss_all = nn.CrossEntropyLoss(reduction='none')(logits.view(-1, 256), y.view(-1)).view(2, -1)
                
                # Focus 100% of the network's capacity on the retrieval task
                loss_passkey = loss_all[:, -5:].mean()
                loss = loss_passkey
                
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            if step % 100 == 0:
                print(f"Step {step:3d} | Loss Passkey: {loss_passkey.item():.4f} | VRAM: {get_gpu_memory_usage():.0f}MB")
                
    print(f"\nCurriculum Training Complete in {time.time() - t_start:.1f}s")
    
    # After training, quick diagnostic test
    model.eval()
    x, _, keys = generate_passkey_data(1, 512, 0.5, device)
    target_key = keys[0]
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, states = model(x)
            gen = x
            next_byte = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            gen = torch.cat([gen, next_byte], dim=1)
            for _ in range(4):
                logits, states = model(next_byte, states)
                next_byte = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                gen = torch.cat([gen, next_byte], dim=1)
    
    print("\n[Diagnostic] Checking if model learned the format:")
    print(f"Target Passkey: {target_key}")
    print("Generated:     ", bytes(gen[0, -5:].tolist()).decode(errors='replace'))
    
    return model

def evaluate_context_window(model):
    device = next(model.parameters()).device
    model.eval()
    
    test_lengths = [1024, 2048, 4096, 8192, 16384]
    test_depths = [0.1, 0.5, 0.9]
    trials = 5
    
    results = []
    
    print("\n--- Evaluating Context Window ---")
    
    for L in test_lengths:
        for depth in test_depths:
            successes = 0
            for _ in range(trials):
                passkey = f"{random.randint(10000, 99999)}"
                passkey_str = f" The passkey is {passkey}. "
                prompt_str = f" What is the passkey? "
                
                fixed_len = len(passkey_str) + len(prompt_str)
                noise_len = max(0, L - fixed_len)
                pos = int(noise_len * depth)
                
                noise1 = torch.randint(97, 123, (pos,), dtype=torch.uint8)
                noise2 = torch.randint(97, 123, (noise_len - pos,), dtype=torch.uint8)
                
                p1 = torch.tensor(list(passkey_str.encode('utf-8')), dtype=torch.uint8)
                p2 = torch.tensor(list(prompt_str.encode('utf-8')), dtype=torch.uint8)
                
                prompt_tensor = torch.cat([noise1, p1, noise2, p2]).long().unsqueeze(0).to(device)
                
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        # Pure argmax greedy decoding to avoid sampling noise
                        generated = prompt_tensor
                        logits, states = model(prompt_tensor)
                        next_byte = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                        generated = torch.cat([generated, next_byte], dim=1)
                        for _ in range(4):
                            logits, next_states = model(next_byte, states)
                            states = next_states
                            next_byte = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                            generated = torch.cat([generated, next_byte], dim=1)
                        
                gen_bytes = generated[0, -5:].cpu().numpy().tolist()
                try:
                    gen_str = bytes(gen_bytes).decode('utf-8')
                except:
                    gen_str = ""
                    
                if gen_str == passkey:
                    successes += 1
                elif _ == 0:
                    # Print the first failure of each depth for debugging
                    print(f"    [Debug] L={L}, Depth={depth} | Target: {passkey} | Generated: {gen_str}")
                    
            acc = successes / trials
            print(f"L={L:5d} | Depth={depth:.1f} | Accuracy: {acc*100:3.0f}% | VRAM: {get_gpu_memory_usage():.0f}MB")
            results.append({"length": L, "depth": depth, "accuracy": acc})
            
    return results

def run_experiment():
    model = train_passkey_curriculum()
    results = evaluate_context_window(model)
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nExperiment 11 Complete. Saved results.json")

if __name__ == "__main__":
    run_experiment()
