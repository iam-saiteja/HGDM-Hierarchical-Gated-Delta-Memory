import torch
import torch.nn as nn
import time
import sys
import os

# Ensure we can import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hgdm_ultimate import HGDMUltimate, HGDMConfig
from hgdm_omega import OmegaGDM, OmegaConfig

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_models():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA is not available. This benchmark must run on the GPU server.")
        return
        
    device = torch.device("cuda")
    print("==============================================================")
    print(f"Running Benchmark on: {torch.cuda.get_device_name(0)}")
    print("==============================================================")
    
    # -------------------------------------------------------------------------
    # 1. Initialize configurations (Matched at standard ~120M base core size)
    # -------------------------------------------------------------------------
    # Previous Monolithic HGDM
    config_hgdm = HGDMConfig(
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_k=64,
        d_v=64,
        d_ff=3072,
        max_position_embeddings=2048,
        vocab_size=256
    )
    
    # New OmegaGDM (Temporal Hourglass)
    config_omega = OmegaConfig(
        d_byte=256,
        catcher_layers=2,
        renderer_layers=2,
        d_model=768,
        core_layers=12,
        n_heads=12,
        d_k=64,
        d_v=64,
        d_ff=3072,
        decimation_rate=8, # W = 8
        max_position_embeddings=2048,
        vocab_size=256
    )
    
    print("Initializing models...")
    model_hgdm = HGDMUltimate(config_hgdm, force_sequential=False).to(device)
    model_omega = OmegaGDM(config_omega, force_sequential=False).to(device)
    
    params_hgdm = count_parameters(model_hgdm)
    params_omega = count_parameters(model_omega)
    
    print(f"HGDM (Previous) Params: {params_hgdm:,}")
    print(f"OmegaGDM (New) Params:   {params_omega:,}")
    print(f"Parameter Ratio (OmegaGDM / HGDM): {params_omega / params_hgdm:.2%}")
    print("--------------------------------------------------------------")
    
    # -------------------------------------------------------------------------
    # Setup test tensors (Standard sequence length = 2048, batch size = 8)
    # -------------------------------------------------------------------------
    B = 8
    T = 2048
    inputs = torch.randint(0, 256, (B, T), device=device)
    
    # Helper to measure forward and backward pass speed and VRAM
    def run_profile(model, name):
        # Warmup
        for _ in range(3):
            out, _ = model(inputs)
            loss = out.mean()
            loss.backward()
            model.zero_grad()
            
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Timing events
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        
        # Measure VRAM & Speed
        start_evt.record()
        
        # Forward pass
        out, _ = model(inputs)
        loss = out.mean()
        # Backward pass
        loss.backward()
        
        end_evt.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_evt.elapsed_time(end_evt)
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        model.zero_grad()
        return elapsed_ms, vram_mb

    # Helper to measure autoregressive generation speed (latency per token)
    def run_generation_profile(model, name, prompt_len=128, gen_len=32):
        prompt = torch.randint(0, 256, (1, prompt_len), device=device)
        
        # Warmup
        _ = model.generate(prompt, max_new_bytes=5, temp=0.8)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        _ = model.generate(prompt, max_new_bytes=gen_len, temp=0.8)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000.0
        ms_per_token = latency_ms / gen_len
        return ms_per_token

    print("Benchmarking HGDM (Previous)...")
    time_hgdm, vram_hgdm = run_profile(model_hgdm, "HGDM")
    gen_speed_hgdm = run_generation_profile(model_hgdm, "HGDM")
    
    print("Benchmarking OmegaGDM (New)...")
    time_omega, vram_omega = run_profile(model_omega, "OmegaGDM")
    gen_speed_omega = run_generation_profile(model_omega, "OmegaGDM")
    
    print("\n================== BENCHMARK RESULTS ==================")
    print(f"Sequence Length: {T} | Batch Size: {B}")
    print("--------------------------------------------------------------")
    print(f"{'Metric':<30} | {'HGDM (Previous)':<18} | {'OmegaGDM (New)':<18} | {'Improvement':<12}")
    print("--------------------------------------------------------------")
    print(f"{'Total Parameters':<30} | {params_hgdm:<18,} | {params_omega:<18,} | {((params_hgdm - params_omega)/params_hgdm)*100:+.2f}%")
    print(f"{'Training Pass (FWD+BWD) Latency':<30} | {time_hgdm:<14.2f} ms | {time_omega:<14.2f} ms | {((time_hgdm - time_omega)/time_hgdm)*100:+.2f}% speedup")
    print(f"{'Peak Training VRAM Allocation':<30} | {vram_hgdm:<15.2f} MB | {vram_omega:<15.2f} MB | {((vram_hgdm - vram_omega)/vram_hgdm)*100:+.2f}% reduction")
    print(f"{'Generation Speed (per token)':<30} | {gen_speed_hgdm:<14.2f} ms | {gen_speed_omega:<14.2f} ms | {((gen_speed_hgdm - gen_speed_omega)/gen_speed_hgdm)*100:+.2f}% speedup")
    print("==============================================================")

if __name__ == "__main__":
    benchmark_models()
