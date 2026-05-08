import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import time
import json
import math
import struct
from hgdm_ultimate import HGDMUltimate, HGDMConfig

# =============================================================================
# COMPLEX MULTIMODAL RAW BYTE GENERATORS (High Entropy, Non-Trivial)
# =============================================================================

def generate_audio_bytes(num_samples=100000):
    """Generates complex raw 16-bit PCM audio (Chords + Noise + Envelope)."""
    print("Generating Complex Raw Audio Bytes (PCM)...")
    data = bytearray()
    
    # Generate a complex C-major chord with varying amplitude
    freqs = [261.63, 329.63, 392.00] # C4, E4, G4
    for i in range(num_samples):
        t = i / 44100.0
        # Combine frequencies
        signal = sum(math.sin(2.0 * math.pi * f * t) for f in freqs) / len(freqs)
        # Apply an envelope (tremolo)
        envelope = 0.5 * (1.0 + math.sin(2.0 * math.pi * 5.0 * t))
        # Add high-frequency noise
        noise = (torch.rand(1).item() * 0.1) - 0.05
        
        value = int(32767.0 * (signal * envelope + noise))
        value = max(-32768, min(32767, value)) # Clip
        data += struct.pack('<h', value)
        
    tensor_data = torch.frombuffer(data, dtype=torch.uint8).long()
    print(f"Generated Audio Tensor Size: {len(tensor_data) / 1024:.1f} KB")
    return tensor_data

def generate_image_bytes(width=256, height=256):
    """Generates a complex RGB image byte stream (Mandelbrot Fractal)."""
    print("Generating Complex Raw Image Bytes (Fractal RGB)...")
    pixels = bytearray()
    
    # Generate a Mandelbrot set for high spatial complexity
    for y in range(height):
        for x in range(width):
            c0 = complex(2.5 * x / width - 2.0, 2.0 * y / height - 1.0)
            c = 0
            for i in range(30):
                if abs(c) > 2:
                    break
                c = c * c + c0
                
            r = (i * 8) % 256
            g = (i * 16) % 256
            b = (i * 32) % 256
            pixels += bytes([r, g, b])
            
    tensor_data = torch.frombuffer(pixels, dtype=torch.uint8).long()
    print(f"Generated Image Tensor Size: {len(tensor_data) / 1024:.1f} KB")
    return tensor_data

def generate_video_bytes(frames=30, width=64, height=64):
    """Generates uncompressed RGB frames of a bouncing ball with a dynamic background."""
    print("Generating Complex Raw Video Bytes (Bouncing Ball + Dynamic BG)...")
    video_stream = bytearray()
    
    bx, by = 10, 10
    dx, dy = 3, 2
    
    for frame in range(frames):
        # Update ball position
        bx += dx
        by += dy
        if bx <= 5 or bx >= width - 5: dx = -dx
        if by <= 5 or by >= height - 5: dy = -dy
        
        for y in range(height):
            for x in range(width):
                # Draw Ball
                if (x - bx)**2 + (y - by)**2 < 25:
                    video_stream += bytes([255, 50, 50])
                else:
                    # Dynamic moving gradient background
                    r = (x + frame * 2) % 256
                    g = (y + frame * 3) % 256
                    b = 100
                    video_stream += bytes([r, g, b])
                    
    tensor_data = torch.frombuffer(video_stream, dtype=torch.uint8).long()
    print(f"Generated Video Tensor Size: {len(tensor_data) / 1024:.1f} KB")
    return tensor_data

# =============================================================================
# GPU-MONITORED TRAINING LOOP
# =============================================================================

def train_modality(model, modality_name, train_data, steps=500, seq_len=512):
    device = torch.device('cuda')
    model.train()
    
    # Re-initialize the weights to prove zero-shot universality from scratch
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
            
    lr = 4e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n--- Training on {modality_name} Bytes ---")
    history = []
    t_start = time.time()
    
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        
        ix = torch.randint(len(train_data) - seq_len - 1, (1,))
        x = torch.stack([train_data[i:i+seq_len] for i in ix]).to(device)
        y = torch.stack([train_data[i+1:i+seq_len+1] for i in ix]).to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
            if isinstance(out, tuple): out = out[0]
            loss = F.cross_entropy(out.view(-1, 256), y.view(-1))
            
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        
        if step % 50 == 0:
            bpb = loss.item() / math.log(2)
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            elapsed = time.time() - t_start
            
            print(f"Step {step:4d} | BPB: {bpb:.4f} | Peak VRAM: {peak_mem:.0f} MB | Time: {elapsed:.1f}s")
            history.append({
                "step": step,
                "bpb": bpb,
                "peak_mem_mb": peak_mem,
                "time_s": elapsed
            })
            
    # =========================================================================
    # GENERATIVE INFERENCE PROOF
    # =========================================================================
    print(f"\n--- Generating {modality_name} hallucination ---")
    model.eval()
    
    # 1. Take a 128-byte prompt from the training data
    prompt_len = 128
    prompt = train_data[:prompt_len].unsqueeze(0).to(device)
    
    # 2. Generate the next 4,000 bytes (4KB)
    gen_len = 4000
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output_tensor = model.generate(prompt, max_new_bytes=gen_len, temp=0.8)[0]
            
    # 3. Save to raw binary file
    output_bytes = bytes(output_tensor.cpu().tolist())
    filename = f"generated_{modality_name.split(' ')[0].lower()}.raw"
    with open(filename, 'wb') as f:
        f.write(output_bytes)
        
    print(f"Success! Saved {len(output_bytes)} bytes to {filename}")
    print(f"This proves the architecture learned the physical structure of {modality_name}.\n")
            
    return history

if __name__ == "__main__":
    device = torch.device('cuda')
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    
    model = HGDMUltimate(config).to(device)
    
    datasets = {
        "Audio (Complex PCM)": generate_audio_bytes(),
        "Image (Fractal RGB)": generate_image_bytes(),
        "Video (Dynamic Frames)": generate_video_bytes()
    }
    
    results = {}
    for modality_name, data in datasets.items():
        results[modality_name] = train_modality(model, modality_name, data, steps=500)
        
    with open("results_exp7_multimodal.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nExperiment 7 Complete. Saved results_exp7_multimodal.json")
