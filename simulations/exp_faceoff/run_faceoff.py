import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import time
import json
from hgdm_ultimate import HGDMUltimate, HGDMConfig

class BaselineTransformer(nn.Module):
    """A standard causal Transformer for baseline comparison."""
    def __init__(self, d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        seq_len = x.size(1)
        # Create causal mask (PyTorch standard)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.fc_out(x)
        
def benchmark():
    device = torch.device('cuda')
    seq_lengths = [512, 1024, 2048, 4096, 8192]
    batch_size = 1
    
    # Initialize Models (~120M parameters)
    hgdm_config = HGDMConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256)
    
    print("Initializing models...")
    models = {
        "HGDM": HGDMUltimate(hgdm_config).to(device),
        "Transformer": BaselineTransformer().to(device)
    }
    
    results = {"seq_lengths": seq_lengths}
    
    for name, model in models.items():
        print(f"\n--- Benchmarking {name} ---")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        results[name] = {"memory": [], "throughput": []}
        
        for seq_len in seq_lengths:
            print(f"Testing seq_len={seq_len}...", end=" ", flush=True)
            try:
                x = torch.randint(0, 256, (batch_size, seq_len)).to(device)
                y = torch.randint(0, 256, (batch_size, seq_len)).to(device)
                
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Warmup (Compile kernels)
                for _ in range(3):
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda'):
                        if name == "Transformer":
                            # DISABLE FLASH ATTENTION to expose raw mathematical scaling
                            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                                out = model(x)
                        else:
                            out = model(x)
                            
                        if isinstance(out, tuple): out = out[0]
                        loss = nn.CrossEntropyLoss()(out.view(-1, 256), y.view(-1))
                    loss.backward()
                    optimizer.step()
                
                # Measure Time
                torch.cuda.synchronize()
                start_time = time.time()
                iters = 10
                for _ in range(iters):
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda'):
                        if name == "Transformer":
                            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                                out = model(x)
                        else:
                            out = model(x)
                            
                        if isinstance(out, tuple): out = out[0]
                        loss = nn.CrossEntropyLoss()(out.view(-1, 256), y.view(-1))
                    loss.backward()
                    optimizer.step()
                torch.cuda.synchronize()
                end_time = time.time()
                
                # Calculate metrics
                peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
                time_per_iter = (end_time - start_time) / iters
                tokens_per_sec = (batch_size * seq_len) / time_per_iter
                
                results[name]["memory"].append(peak_mem)
                results[name]["throughput"].append(tokens_per_sec)
                print(f"Mem: {peak_mem:.0f}MB, Speed: {tokens_per_sec:.0f} tok/s")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("OOM (Out of Memory)!")
                    results[name]["memory"].append(None)
                    results[name]["throughput"].append(None)
                    torch.cuda.empty_cache()
                else:
                    raise e
                    
    with open("faceoff_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved benchmark results to faceoff_results.json")

if __name__ == "__main__":
    benchmark()
