import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import time
import random
import json
from hgdm_ultimate import HGDMUltimate, HGDMConfig
from utils import get_gpu_memory_usage

def generate_copy_data(batch_size, seq_len, device):
    """
    Generates Selective Copy data.
    Pattern (10 random digits) + Separator (':') + Noise (random lowercase) + Trigger ('=') + Pattern
    """
    x_batch = []
    y_batch = []
    
    for _ in range(batch_size):
        # 10 random digits as the pattern to copy
        pattern = torch.randint(48, 58, (10,), dtype=torch.uint8)
        
        # Noise: random lowercase letters
        noise_len = max(0, seq_len - 12) # 10 pattern + 2 markers
        noise = torch.randint(97, 123, (noise_len,), dtype=torch.uint8)
        
        sep = torch.tensor([58], dtype=torch.uint8) # ':'
        trig = torch.tensor([61], dtype=torch.uint8) # '='
        
        seq = torch.cat([pattern, sep, noise, trig, pattern])
        x_batch.append(seq[:-1])
        y_batch.append(seq[1:])
        
    x = torch.stack(x_batch).long().to(device)
    y = torch.stack(y_batch).long().to(device)
    return x, y

def train_copy_curriculum():
    device = torch.device('cuda')
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    model = HGDMUltimate(config).to(device)
    
    checkpoint_path = "../exp1_enwik8/hgdm_enwik8_120M.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading base Enwik8 checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Handle older checkpoint naming conventions
        if "byte_emb.weight" in state_dict:
            state_dict["embedding.weight"] = state_dict.pop("byte_emb.weight")
        if "head.weight" in state_dict:
            state_dict["fc_out.weight"] = state_dict.pop("head.weight")
        model.load_state_dict(state_dict, strict=False)
        # The base Enwik8 checkpoint was trained without pos_embedding.
        with torch.no_grad():
            model.pos_embedding.zero_()
    else:
        print("WARNING: Checkpoint not found. Training from scratch will fail to learn retrieval.")
    
    # Lowered LR to 5e-5 for stable fine-tuning
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n{'='*50}\nExp 11: Selective Copy (Context Window Test)\n{'='*50}")
    
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
            x, y = generate_copy_data(2, seq_len, device)
            
            opt.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(x)
                loss_all = nn.CrossEntropyLoss(reduction='none')(logits.view(-1, 256), y.view(-1)).view(2, -1)
                
                # Include full sequence loss to stabilize gradients, plus a boost for the 10-byte pattern
                loss_pattern = loss_all[:, -10:].mean()
                loss_seq = loss_all.mean()
                loss = loss_seq + loss_pattern * 2.0
                
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            if step % 100 == 0:
                print(f"Step {step:3d} | Loss Pattern: {loss_pattern.item():.4f} | VRAM: {get_gpu_memory_usage():.0f}MB")
                
    print(f"\nCurriculum Training Complete in {time.time() - t_start:.1f}s")
    
    # After training, quick diagnostic test
    model.eval()
    x, y = generate_copy_data(1, 512, device)
    
    target_pattern = bytes(y[0, -10:].tolist()).decode('utf-8', errors='replace')
    prompt_tensor = x[:, :-10]
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, states = model(prompt_tensor)
            gen = prompt_tensor
            next_byte = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            gen = torch.cat([gen, next_byte], dim=1)
            for _ in range(9):
                logits, next_states = model(next_byte, states)
                states = next_states
                next_byte = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                gen = torch.cat([gen, next_byte], dim=1)
    
    gen_pattern = bytes(gen[0, -10:].tolist()).decode('utf-8', errors='replace')
    
    print("\n[Diagnostic] Checking if model learned the format:")
    print(f"Target Pattern: {target_pattern}")
    print(f"Generated:      {gen_pattern}")
    
    return model

def evaluate_context_window(model):
    device = next(model.parameters()).device
    model.eval()
    
    test_lengths = [1024, 2048, 4096, 8192, 16384]
    trials = 5
    
    results = []
    
    print("\n--- Evaluating Context Window (Selective Copy) ---")
    
    for L in test_lengths:
        successes = 0
        for _ in range(trials):
            x, y = generate_copy_data(1, L, device)
            prompt_tensor = x[:, :-10]
            target_list = y[0, -10:].tolist()
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    generated = prompt_tensor
                    logits, states = model(prompt_tensor)
                    next_byte = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_byte], dim=1)
                    for _ in range(9):
                        logits, next_states = model(next_byte, states)
                        states = next_states
                        next_byte = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                        generated = torch.cat([generated, next_byte], dim=1)
                        
            gen_list = generated[0, -10:].cpu().numpy().tolist()
            
            if gen_list == target_list:
                successes += 1
            elif _ == 0:
                target_str = bytes(target_list).decode('utf-8', errors='replace')
                gen_str = bytes(gen_list).decode('utf-8', errors='replace')
                print(f"    [Debug] L={L} | Target: {target_str} | Generated: {gen_str}")
                
        acc = successes / trials
        print(f"L={L:5d} | Accuracy: {acc*100:3.0f}% | VRAM: {get_gpu_memory_usage():.0f}MB")
        results.append({"seq_len": L, "accuracy": acc})
        
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nExperiment 11 Complete. Saved results.json")

def run_experiment():
    model = train_copy_curriculum()
    evaluate_context_window(model)

if __name__ == "__main__":
    run_experiment()
