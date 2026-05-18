import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
import subprocess
import urllib.request
import zipfile
from hgdm_ultimate import HGDMUltimate, HGDMConfig

# =============================================================================
# UTILITIES
# =============================================================================
def get_gpu_stats():
    try:
        cmd = "nvidia-smi --query-gpu=temperature.gpu,memory.used,utilization.gpu --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        temp, mem, util = output.split(',')
        return f"{temp.strip()}C | {mem.strip()}MB | {util.strip()}% Util"
    except:
        return "N/A"

def get_temp():
    try:
        cmd = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        return int(output)
    except:
        return 0

def get_data():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "enwik8.zip")
    data_path = os.path.join(data_dir, "enwik8")
    
    if not os.path.exists(data_path):
        if not os.path.exists(zip_path):
            print("Downloading enwik8 (100MB)... This may take a minute.")
            url = "http://mattmahoney.net/dc/enwik8.zip"
            try:
                urllib.request.urlretrieve(url, zip_path)
            except Exception as e:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                raise RuntimeError(f"Failed to download enwik8 from {url}. Error: {e}")
        
        print("Extracting enwik8...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
        except zipfile.BadZipFile:
            print(f"Error: {zip_path} is corrupted (BadZipFile). Deleting it so you can retry.")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise RuntimeError(
                "Downloaded zip file was corrupted or incomplete (often happens when Matt Mahoney's site has a transient error or rate limit).\n"
                "We deleted the bad zip file. Please run the script again to auto-retry, or download it manually with:\n"
                "  wget -P data/ http://mattmahoney.net/dc/enwik8.zip\n"
                "and then run this script again!"
            )
            
    with open(data_path, 'rb') as f:
        data = f.read()
    
    n = len(data)
    train_data = torch.frombuffer(data[:int(n * 0.9)], dtype=torch.uint8).long()
    val_data = torch.frombuffer(data[int(n * 0.9):], dtype=torch.uint8).long()
    return train_data, val_data

# =============================================================================
# TRANSFORMER BASELINE (For Comparison)
# =============================================================================
class KVCachedAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        if kv_cache is not None:
            pk, pv = kv_cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
            
        new_kv_cache = (k, v)
        is_causal = (kv_cache is None) and (T > 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), new_kv_cache

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = KVCachedAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, kv_cache=None):
        nx = self.norm1(x)
        a_out, new_cache = self.attn(nx, kv_cache)
        x = x + a_out
        x = x + self.mlp(self.norm2(x))
        return x, new_cache

class TransformerBaseline(nn.Module):
    def __init__(self, d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256, max_seq_len=20000):
        super().__init__()
        self.byte_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([TransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, kv_caches=None):
        B, T = x.shape
        if kv_caches is None:
            pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        else:
            past_seq_len = kv_caches[0][0].shape[2]
            pos = torch.arange(past_seq_len, past_seq_len + T, device=x.device).unsqueeze(0)
            
        x = self.byte_emb(x) + self.pos_emb(pos)
        
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, nc = layer(x, cache_i)
            new_caches.append(nc)
            
        return self.head(self.norm_f(x)), new_caches

    @torch.no_grad()
    def generate(self, prompt, max_new_bytes=100, temp=0.8):
        self.eval()
        generated = prompt
        logits, kv_caches = self.forward(prompt)
        last_byte = prompt[:, -1:]
        
        for _ in range(max_new_bytes):
            logits, kv_caches = self.forward(last_byte, kv_caches)
            logits = logits[:, -1, :] / temp
            probs = F.softmax(logits, dim=-1)
            last_byte = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, last_byte], dim=1)
        return generated

# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================
def train_model(model, name, train_data, steps=1000, micro_batch=1, accum_steps=12, seq_len=2048, lr=4e-4):
    device = torch.device('cuda')
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Warmup + Cosine Decay Scheduler for stability
    warmup_steps = 100
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps - warmup_steps, eta_min=lr/10)
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])
    scaler = torch.amp.GradScaler('cuda')
    
    train_data = train_data.to(device)
    print(f"\n{'='*60}\nPRODUCTION RUN: {name} on Enwik8\n{'='*60}")
    print(f"Config: Steps={steps}, Context={seq_len}, Initial LR={lr}")
    
    history = []
    t_start = time.time()
    t_last = time.time()
    
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        try:
            for _ in range(accum_steps):
                ix = torch.randint(len(train_data) - seq_len - 1, (micro_batch,))
                x = torch.stack([train_data[i:i+seq_len] for i in ix])
                y = torch.stack([train_data[i+1:i+seq_len+1] for i in ix])
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)[0]
                    loss = F.cross_entropy(logits.view(-1, 256), y.view(-1)) / accum_steps
                    
                scaler.scale(loss).backward()
                accum_loss += loss.item() * accum_steps
                
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step() # Advance LR decay
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[{name}] CRITICAL OOM ERROR at Step {step}: {e}")
                torch.cuda.empty_cache()
                return history, time.time() - t_start
            else:
                raise e
        
        if step % 50 == 0:
            avg_step_loss = accum_loss / accum_steps
            bpb = avg_step_loss / math.log(2)
            elapsed = time.time() - t_last
            gpu = get_gpu_stats()
            print(f"Step {step:4d} | BPB: {bpb:.4f} | Time: {elapsed:.1f}s | GPU: {gpu}")
            history.append((step, bpb, elapsed, gpu))
            t_last = time.time()
            
            # THERMAL THROTTLE: Option 3 (Strong Control - Checked every 50 steps)
            try:
                curr_temp = int(gpu.split('C')[0])
                if curr_temp > 85:
                    print(f"[THERMAL CONTROL] GPU hit {curr_temp}C. Sleeping 20 seconds to cool down...")
                    time.sleep(20)
            except:
                pass
            
            if step > 0 and step % 500 == 0:
                torch.save(model.state_dict(), f"{name.lower()}_enwik8.pt")
                print(f"--> Checkpoint saved at step {step}")
            
    total_time = time.time() - t_start
    torch.save(model.state_dict(), f"{name.lower()}_enwik8.pt")
    return history, total_time

@torch.no_grad()
def evaluate_model(model, val_data, seq_len=2048, batch_size=4, batches=50):
    device = next(model.parameters()).device
    model.eval()
    val_data = val_data.to(device)
    total_loss = 0
    
    try:
        for _ in range(batches):
            ix = torch.randint(len(val_data) - seq_len - 1, (batch_size,))
            x = torch.stack([val_data[i:i+seq_len] for i in ix])
            y = torch.stack([val_data[i+1:i+seq_len+1] for i in ix])
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)[0]
                loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
                
            total_loss += loss.item()
            
        avg_loss = total_loss / batches
        return avg_loss / math.log(2), math.exp(avg_loss)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Evaluation failed due to OOM: {e}")
            torch.cuda.empty_cache()
            return float('inf'), float('inf')
        raise e

def benchmark_inference(model, lengths=[100, 250, 500, 1000, 2000, 5000, 10000]):
    device = next(model.parameters()).device
    model.eval()
    prompt = torch.tensor([[ord('R')]], dtype=torch.long, device=device)
    
    try:
        _ = model.generate(prompt, max_new_bytes=10)
    except:
        pass # Ignore warmup OOM
    
    times = {}
    mems = {}
    for length in lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        
        try:
            _ = model.generate(prompt, max_new_bytes=length)
            torch.cuda.synchronize()
            times[length] = time.time() - t0
            mems[length] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except RuntimeError as e:
            print(f"Generation failed at {length} bytes: {e}")
            times[length] = float('inf')
            mems[length] = 0
            
    return times, mems

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-hgdm", action="store_true", help="Skip training the Transformer baseline and just load it for validation/generation.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading Enwik8 (100MB)...")
    train_data, val_data = get_data()
    print(f"Train Size: {len(train_data)/1024**2:.1f}MB | Val Size: {len(val_data)/1024**2:.1f}MB")
    
    # ---------------------------------------------------------
    # 1. INIT MODELS
    # ---------------------------------------------------------
    config = HGDMConfig(
        d_model=768,       
        n_layers=12,       
        n_heads=12,        
        d_k=64,            
        d_v=64,            
        d_ff=3072,         
        vocab_size=256
    )
    hgdm_model = HGDMUltimate(config).to(device)
    
    # Matching Transformer Config (same width and depth)
    tf_model = TransformerBaseline(
        d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256, max_seq_len=20000
    ).to(device)
    
    print(f"\nHGDM Parameters: {sum(p.numel() for p in hgdm_model.parameters()) / 1e6:.2f} M")
    print(f"Transformer Parameters: {sum(p.numel() for p in tf_model.parameters()) / 1e6:.2f} M")
    
    tf_history, hgdm_history = [], []
    tf_time, hgdm_time = 0.0, 0.0
    
    # ---------------------------------------------------------
    # 2. TRAIN TRANSFORMER
    # ---------------------------------------------------------
    if os.path.exists("transformer_enwik8.pt"):
        print("\nLoading existing Transformer weights...")
        tf_model.load_state_dict(torch.load("transformer_enwik8.pt", map_location=device, weights_only=True))
        
    if args.only_hgdm:
        print("Skipping Transformer training (--only-hgdm is active).")
        tf_time = 555.7 # Default recorded baseline
    else:
        print("Training Transformer...")
        tf_history, tf_time = train_model(tf_model, "Transformer", train_data)
        
    # ---------------------------------------------------------
    # 3. TRAIN HGDM
    # ---------------------------------------------------------
    torch.manual_seed(42)
    if os.path.exists("hgdm_ultimate_enwik8.pt"):
        print("\nLoading existing HGDM weights for continued training...")
        hgdm_model.load_state_dict(torch.load("hgdm_ultimate_enwik8.pt", map_location=device, weights_only=True))
    
    # V6 Comparison Run (1000 steps, 2048 Context)
    hgdm_history, hgdm_time = train_model(
        hgdm_model, "HGDMUltimate", train_data, 
        steps=1000, seq_len=2048, lr=4e-4
    )
        
    # ---------------------------------------------------------
    # 4. EVALUATE & BENCHMARK
    # ---------------------------------------------------------
    print("\nEvaluating Transformer on Validation Set...")
    tf_bpb, tf_ppl = evaluate_model(tf_model, val_data)
    
    print("\nEvaluating HGDM on Validation Set...")
    hgdm_bpb, hgdm_ppl = evaluate_model(hgdm_model, val_data)
    
    print("\nBenchmarking Inference Speeds & Memory...")
    lengths = [100, 250, 500, 1000, 2000, 5000, 10000]
    
    print("Testing Transformer...")
    tf_times, tf_mems = benchmark_inference(tf_model, lengths)
        
    print("Testing HGDM...")
    hgdm_times, hgdm_mems = benchmark_inference(hgdm_model, lengths)
    
    print("\nGenerating Text Samples...")
    prompt = torch.tensor([list("Wikipedia is ".encode('utf-8'))], dtype=torch.long, device=device)
    
    torch.manual_seed(42)
    try:
        tf_text = bytes(tf_model.generate(prompt, max_new_bytes=500)[0].tolist()).decode('utf-8', errors='replace')
    except:
        tf_text = "FAILED (OOM)"
        
    torch.manual_seed(42)
    hgdm_text = bytes(hgdm_model.generate(prompt, max_new_bytes=500)[0].tolist()).decode('utf-8', errors='replace')
    
    print("\nCompiling Final Report...")
    with open("ultimate_enwik8_results.md", "w") as f:
        f.write("# 🚀 HGDMUltimate vs Transformer on Enwik8 (100MB)\n\n")
        
        f.write("## 1. Validation Performance\n")
        f.write("| Model | Parameters | Val BPB | Val Perplexity | Total Training Time |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        t_bpb_str = f"{tf_bpb:.4f}" if tf_bpb != float('inf') else "OOM"
        t_ppl_str = f"{tf_ppl:.2f}" if tf_ppl != float('inf') else "OOM"
        f.write(f"| **Transformer** | ~{sum(p.numel() for p in tf_model.parameters())/1e6:.1f}M | {t_bpb_str} | {t_ppl_str} | {tf_time:.1f}s |\n")
        f.write(f"| **HGDMUltimate** | ~{sum(p.numel() for p in hgdm_model.parameters())/1e6:.1f}M | **{hgdm_bpb:.4f}** | **{hgdm_ppl:.2f}** | {hgdm_time:.1f}s |\n\n")
        
        f.write("## 2. Inference Speed & Memory Scaling Benchmark\n")
        f.write("| Sequence Length | TF Speed | HGDM Speed | TF VRAM | HGDM VRAM |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for L in lengths:
            tf_bps_str = f"{L / tf_times[L]:.0f} B/s" if tf_times[L] != float('inf') else "OOM"
            hg_bps_str = f"{L / hgdm_times[L]:.0f} B/s" if hgdm_times[L] != float('inf') else "OOM"
            tf_vram_str = f"{tf_mems[L]:.0f} MB" if tf_mems[L] > 0 else "OOM"
            f.write(f"| {L} bytes | {tf_bps_str} | {hg_bps_str} | {tf_vram_str} | {hgdm_mems[L]:.0f} MB |\n")
        f.write("\n")
        
        f.write("## 3. Generative Quality (Sample Output)\n")
        f.write("**Prompt:** `Wikipedia is `\n\n")
        f.write("### Transformer\n```text\n" + tf_text + "\n```\n\n")
        f.write("### HGDMUltimate\n```text\n" + hgdm_text + "\n```\n\n")
        
        if tf_history or hgdm_history:
            f.write("## 4. Training Convergence Log\n")
            f.write("| Step | TF Train BPB | TF GPU Mem | HGDM Train BPB | HGDM GPU Mem |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            max_len = max(len(tf_history), len(hgdm_history))
            for i in range(max_len):
                t_bpb = tf_history[i][1] if i < len(tf_history) else ("Skipped" if args.only_hgdm else "OOM")
                t_mem = tf_history[i][3].split('|')[1].strip() if i < len(tf_history) else ("Skipped" if args.only_hgdm else "OOM")
                h_bpb = hgdm_history[i][1] if i < len(hgdm_history) else "N/A"
                h_mem = hgdm_history[i][3].split('|')[1].strip() if i < len(hgdm_history) else "N/A"
                
                step = tf_history[i][0] if i < len(tf_history) else (hgdm_history[i][0] if i < len(hgdm_history) else i*50)
                if isinstance(t_bpb, float): t_bpb = f"{t_bpb:.4f}"
                if isinstance(h_bpb, float): h_bpb = f"{h_bpb:.4f}"
                f.write(f"| {step} | {t_bpb} | {t_mem} | {h_bpb} | {h_mem} |\n")

    print(f"\n{'='*60}\nALL RUNS COMPLETE\n{'='*60}")
    print("Everything has been merged and logged into: ultimate_enwik8_results.md")
