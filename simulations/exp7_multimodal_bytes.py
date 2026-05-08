import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import time
import json
import math
import struct
import wave
from hgdm_ultimate import HGDMUltimate, HGDMConfig

# =============================================================================
# SYNTHETIC MULTIMODAL RAW BYTE GENERATORS (No Container Formats)
# =============================================================================

def generate_audio_bytes(num_samples=100000):
    """Generates raw 16-bit PCM audio bytes (a 440Hz sine wave). No WAV header."""
    print("Generating Raw Audio Bytes (PCM)...")
    freq = 440.0
    data = bytearray()
    for i in range(num_samples):
        # 16-bit signed integer PCM
        value = int(32767.0 * math.sin(2.0 * math.pi * freq * (i / 44100.0)))
        data += struct.pack('<h', value)
        
    tensor_data = torch.frombuffer(data, dtype=torch.uint8).long()
    print(f"Generated Raw Audio Tensor Size: {len(tensor_data) / 1024:.1f} KB")
    return tensor_data

def generate_image_bytes(width=256, height=256):
    """Generates raw RGB image bytes (a color gradient). No BMP/JPG header."""
    print("Generating Raw Image Bytes (RGB)...")
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            r = int((x / width) * 255)
            g = int((y / height) * 255)
            b = 128
            pixels += bytes([r, g, b])
            
    tensor_data = torch.frombuffer(pixels, dtype=torch.uint8).long()
    print(f"Generated Raw Image Tensor Size: {len(tensor_data) / 1024:.1f} KB")
    return tensor_data

def generate_video_bytes(frames=30, width=64, height=64):
    """Generates a sequence of uncompressed RGB image frames (raw video stream)."""
    print("Generating Raw Video Bytes (Frame Sequence)...")
    video_stream = bytearray()
    
    for frame in range(frames):
        # A moving square
        for y in range(height):
            for x in range(width):
                if frame < x < frame + 10 and frame < y < frame + 10:
                    video_stream += bytes([255, 255, 255])
                else:
                    video_stream += bytes([0, 0, 0])
                    
    tensor_data = torch.frombuffer(video_stream, dtype=torch.uint8).long()
    print(f"Generated Raw Video Tensor Size: {len(tensor_data) / 1024:.1f} KB")
    return tensor_data

# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_modality(model, modality_name, train_data, steps=300, seq_len=512):
    device = torch.device('cuda')
    model.train()
    
    # We use a completely untrained model for each modality to prove zero-shot universality
    # Re-initialize the weights to ensure a fair test per modality
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
            
    lr = 4e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n--- Training on {modality_name} Bytes ---")
    history = []
    
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        
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
            print(f"Step {step:4d} | BPB: {bpb:.4f}")
            history.append({"step": step, "bpb": bpb})
            
    return history

if __name__ == "__main__":
    device = torch.device('cuda')
    config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    
    # We use a single instance of the model, but the training function resets the weights
    # to guarantee we aren't transferring knowledge. It proves the ARCHITECTURE learns.
    model = HGDMUltimate(config).to(device)
    
    datasets = {
        "Audio (WAV)": generate_audio_bytes(),
        "Image (BMP)": generate_image_bytes(),
        "Video (Raw Frames)": generate_video_bytes()
    }
    
    results = {}
    for modality_name, data in datasets.items():
        results[modality_name] = train_modality(model, modality_name, data, steps=500)
        
    with open("results_exp7_multimodal.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nExperiment 7 Complete. Saved results_exp7_multimodal.json")
