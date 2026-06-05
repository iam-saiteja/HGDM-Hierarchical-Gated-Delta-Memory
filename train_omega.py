import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import subprocess
import math
import json
import sys
import os
import argparse

# Explicitly add current directory to path to fix module resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import OmegaGDM
from hgdm_omega import OmegaGDM, OmegaConfig

# Import dataloader from the new one_billion folder
from one_billion.data_1b import get_1b_dataloader

# =============================================================================
# OPTIMIZED STANDARD TRANSFORMER BASELINE (LLaMA-3 Style)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.norm1 = RMSNorm(d_model)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        self.norm2 = RMSNorm(d_model)
        # SwiGLU FFN: 8/3 * d_model
        hidden_dim = int(8 * d_model / 3)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        
        hx = self.norm1(x)
        q = self.wq(hx).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(hx).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(hx).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # RoPE Application
        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)
        
        # FlashAttention (SDPA)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.wo(attn_out)
        
        # SwiGLU FFN
        hx = self.norm2(x)
        swish = F.silu(self.w1(hx))
        hx = swish * self.w3(hx)
        x = x + self.w2(hx)
        
        return x
        
    def rotate_half(self, x):
        x1 = x[..., :self.head_dim//2]
        x2 = x[..., self.head_dim//2:]
        return torch.cat((-x2, x1), dim=-1)

class TopTransformer(nn.Module):
    def __init__(self, vocab_size=256, d_model=1024, n_layers=12, n_heads=16, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerLayer(d_model, n_heads) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)
        self.fc_out.weight = self.embed.weight
        
        # Precompute RoPE frequencies
        theta = 10000.0 ** (-2.0 * (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        m = torch.arange(max_seq_len)
        freqs = torch.outer(m, theta)
        freqs_cis = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", freqs_cis.cos()[None, None, :, :])
        self.register_buffer("sin", freqs_cis.sin()[None, None, :, :])

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x)
        cos = self.cos[:, :, :T, :]
        sin = self.sin[:, :, :T, :]
        
        for layer in self.layers:
            h = layer(h, cos, sin)
            
        h = self.norm(h)
        logits = self.fc_out(h)
        return logits, None
        
    @torch.no_grad()
    def generate(self, prompt_bytes, max_new_bytes=100, temp=0.8):
        self.eval()
        generated = prompt_bytes
        for _ in range(max_new_bytes):
            logits, _ = self.forward(generated)
            next_logit = logits[:, -1, :] / temp
            next_byte = torch.multinomial(F.softmax(next_logit, dim=-1), num_samples=1)
            generated = torch.cat([generated, next_byte], dim=1)
        return generated

# =============================================================================
# TRAINING SCRIPT
# =============================================================================

# [STEP-16] Self-Organizing Curriculum: loss-weighted sampling
class SelfOrganizingCurriculum:
    """Tracks per-document EMA loss and samples harder documents more often."""
    def __init__(self, n_docs, alpha=0.1, temp=2.0):
        self.n_docs = n_docs
        self.alpha = alpha        # EMA smoothing factor
        self.temp = temp          # temperature for softmax sampling
        self.ema_losses = torch.ones(n_docs)  # start uniform
        self._sampling_probs = None
        self._dirty = True

    def update(self, doc_idx, loss_val):
        """Update EMA loss for a document."""
        self.ema_losses[doc_idx] = (1 - self.alpha) * self.ema_losses[doc_idx] + self.alpha * loss_val
        self._dirty = True

    def _recompute_probs(self):
        """Recompute sampling probabilities from EMA losses."""
        logits = self.ema_losses * self.temp
        self._sampling_probs = F.softmax(logits, dim=0)
        self._dirty = False

    @property
    def probs(self):
        if self._dirty:
            self._recompute_probs()
        return self._sampling_probs

    def sample(self, k=1):
        """Sample k document indices weighted by loss."""
        if self._dirty:
            self._recompute_probs()
        return torch.multinomial(self._sampling_probs, k, replacement=True).tolist()

    def kl_vs_uniform(self):
        """KL divergence of current distribution vs uniform."""
        p = self.probs
        uniform = torch.ones_like(p) / self.n_docs
        return (p * (p / uniform).log()).sum().item()


# [STEP-17] Dream / Generative Replay: scheduled generative consolidation
class DreamScheduler:
    """Manages dreaming schedule: fires every `interval` steps after `warmup` steps."""
    def __init__(self, warmup=2000, interval=500, dream_len=32, quality_threshold=2.0, lambda_dream=0.1):
        self.warmup = warmup
        self.interval = interval
        self.dream_len = dream_len
        self.quality_threshold = quality_threshold
        self.lambda_dream = lambda_dream
        self.recent_train_ppl = float('inf')  # EMA of recent training perplexity
        self.ppl_alpha = 0.1

    def update_train_ppl(self, loss):
        """Update EMA of training perplexity from loss."""
        ppl = math.exp(min(loss, 20.0))  # cap to avoid overflow
        if self.recent_train_ppl == float('inf'):
            self.recent_train_ppl = ppl
        else:
            self.recent_train_ppl = (1 - self.ppl_alpha) * self.recent_train_ppl + self.ppl_alpha * ppl

    def should_dream(self, step):
        """Check if dreaming should fire at this step."""
        if step < self.warmup:
            return False
        return (step - self.warmup) % self.interval == 0

    def dream(self, model, device, dtype):
        """Generate a dream sequence and compute consistency loss.
        Returns (dream_loss, dream_ppl, accepted) or (None, None, False) if rejected."""
        model.eval()
        with torch.no_grad():
            seed = torch.randint(0, 256, (1, 1), device=device)
            dream_seq = model.generate(seed, max_new_bytes=self.dream_len, temp=0.5)

        # Compute perplexity of dreamed sequence
        model.train()
        x = dream_seq[:, :-1]
        y = dream_seq[:, 1:]
        if x.shape[1] == 0:
            return None, None, False

        logits, _ = model(x)
        dream_loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        dream_ppl = math.exp(min(dream_loss.item(), 20.0))

        # Quality gate: reject if dream PPL > threshold × recent train PPL
        threshold = self.quality_threshold * self.recent_train_ppl
        if dream_ppl > threshold:
            return None, dream_ppl, False

        # Scale dream loss
        scaled_loss = dream_loss * self.lambda_dream
        return scaled_loss, dream_ppl, True

def get_gpu_memory():
    try:
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        return int(subprocess.check_output(cmd, shell=True).decode().strip())
    except Exception:
        return -1

def get_gpu_temp():
    try:
        cmd = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits"
        return subprocess.check_output(cmd, shell=True).decode().strip() + "C"
    except Exception:
        return "N/A"

def verify_datasets():
    from datasets import load_dataset
    print("[Dataset] Running dataset split pre-start verification...")
    try:
        fw = next(iter(load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)))
        print(f"[Dataset] FineWeb-Edu verified! Sample text len: {len(fw.get('text', ''))}")
        wiki = next(iter(load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)))
        print(f"[Dataset] Wikipedia verified! Sample title: {wiki.get('title', 'N/A')}")
        code = next(iter(load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)))
        print(f"[Dataset] CodeParrot-clean verified! Sample content len: {len(code.get('content', ''))}")
        print("[Dataset] All streaming pipelines verified!")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Dataset verification failed: {e}")
        sys.exit(1)

PROMPTS = [
    "The theory of relativity states that",
    "In machine learning, a transformer model",
    "def fibonacci(n):\n    ",
    "The capital of France is Paris. The capital of Germany is",
]

def run_generation_test(model, device, name, max_new_bytes=150, temp=0.8):
    model.eval()
    print(f"\n{'='*60}")
    print(f"GENERATION TEST: {name}")
    print(f"{'='*60}")
    for i, prompt_text in enumerate(PROMPTS):
        prompt_bytes = list(prompt_text.encode('utf-8', errors='ignore'))
        prompt_tensor = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
        try:
            with torch.no_grad():
                generated = model.generate(prompt_tensor, max_new_bytes=max_new_bytes, temp=temp)
            new_bytes = generated[0, len(prompt_bytes):].tolist()
            decoded = bytes(new_bytes).decode('utf-8', errors='replace')
        except Exception as e:
            decoded = f"[ERROR: {e}]"
        print(f"\n--- Prompt {i+1} ---")
        print(f"PROMPT : {prompt_text!r}")
        print(f"OUTPUT : {decoded!r}")
        sys.stdout.flush()
    model.train()

def train_model(model, name, max_steps, block_size, batch_size, grad_accum, device):
    params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"TRAINING: {name}")
    print(f"{'='*60}")
    print(f"[{name}] Parameters: {params/1e6:.3f} Million")
    
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=1e-5)

    dataloader = get_1b_dataloader(block_size=block_size, batch_size=batch_size)
    data_stream = iter(dataloader)

    model.train()
    logs = []
    t_start = time.time()

    print(f"\n{'Step':<5} | {'Loss':<10} | {'BPB':<7} | {'Net VRAM':<10} | {'StepTime':<9} | {'Elapsed'}")
    print("-" * 65)
    sys.stdout.flush()

    vram_cache = get_gpu_memory()

    for step in range(max_steps):
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        t_step = time.time()

        for _ in range(grad_accum):
            batch = next(data_stream).to(device)
            x, y = batch[:, :-1], batch[:, 1:]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                if isinstance(model, OmegaGDM):
                    logits_list, _ = model(x, return_mtp=True)
                    loss = 0.0
                    for k, logits in enumerate(logits_list):
                        if k == 0:
                            loss = loss + F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
                        else:
                            loss = loss + F.cross_entropy(logits[:, :-k].reshape(-1, 256), y[:, k:].reshape(-1))
                    loss = loss / grad_accum
                else:
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)) / grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[ERROR] NaN/Inf loss at step {step}.")
                break

            loss.backward()
            accum_loss += loss.item() * grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        step_time = time.time() - t_step
        bpb = accum_loss / math.log(2)

        if step % 5 == 0:
            vram_cache = get_gpu_memory()

        logs.append({
            "step": step,
            "loss": accum_loss,
            "bpb": bpb,
            "vram_mb": vram_cache,
            "step_time": step_time,
        })

        if step % 25 == 0 or step == max_steps - 1:
            elapsed = (time.time() - t_start) / 60
            temp = get_gpu_temp()
            print(f"{step:04d} | {accum_loss:<10.4f} | {bpb:<7.4f} | {vram_cache:<6} MB   | {step_time:.2f}s     | {elapsed:.1f}min  ({temp})")
            sys.stdout.flush()

    run_generation_test(model, device, name)
    return logs, params

def main():
    parser = argparse.ArgumentParser(description="Train OmegaGDM vs Top Transformer")
    parser.add_argument("--no-precheck", action="store_true", help="Skip dataset streaming pre-verification")
    parser.add_argument("--only-omega", action="store_true", help="Train only the OmegaGDM architecture, skipping Transformer baseline")
    parser.add_argument("--steps", type=int, default=500, help="Number of training steps")
    parser.add_argument("--quick", action="store_true", help="Train a fast, small model under 10 minutes to verify internals and print interpretability data")
    args = parser.parse_args()

    device = torch.device('cuda')
    assert torch.cuda.is_available(), "CUDA not found."

    if not args.no_precheck:
        verify_datasets()
    else:
        print("[Dataset] Skipping dataset pre-verification check as requested.")

    if args.quick:
        print("[Quick Mode] Activating ultra-fast training configuration (under 2 minutes)...")
        max_steps = 150
        args.only_omega = True
        grad_accum = 2
        batch_size = 4
        block_size = 512
    else:
        max_steps = args.steps
        grad_accum = 4
        batch_size = 8
        block_size = 2048

    # Model 1: LLaMA-3 Style Standard Transformer
    if not args.only_omega:
        transformer_model = TopTransformer(
            vocab_size=256,
            d_model=1024,
            n_layers=12,
            n_heads=16,
            max_seq_len=2048
        ).to(device)
        
        logs_trans, params_trans = train_model(
            transformer_model, "Top Transformer (Baseline)", max_steps, block_size, batch_size, grad_accum, device
        )
        
        del transformer_model
        torch.cuda.empty_cache()
    else:
        logs_trans, params_trans = None, None

    # Model 2: OmegaGDM V2.1
    if args.quick:
        omega_cfg = OmegaConfig(
            d_byte=256,
            catcher_layers=1,
            renderer_layers=1,
            d_model=256,
            core_layers=4,
            n_heads=4,
            d_k=32,
            d_v=32,
            d_ff=512,
            decimation_rate=8,
            max_position_embeddings=2048,
            vocab_size=256,
            use_state_fusion=False
        )
    else:
        omega_cfg = OmegaConfig(
            d_byte=256,
            catcher_layers=2,
            renderer_layers=2,
            d_model=1024,
            core_layers=12,
            n_heads=16,
            d_k=64,
            d_v=64,
            d_ff=4096,
            decimation_rate=8,
            max_position_embeddings=2048,
            vocab_size=256,
            use_state_fusion=False
        )
    omega_model = OmegaGDM(omega_cfg, force_sequential=False).to(device)
    
    logs_omega, params_omega = train_model(
        omega_model, "OmegaGDM (New)", max_steps, block_size, batch_size, grad_accum, device
    )

    # Save checkpoint
    ckpt_path = "omega_checkpoint.pt"
    torch.save(omega_model.state_dict(), ckpt_path)
    print(f"\n[Training] Saved trained model checkpoint to: {ckpt_path}")

    # FINAL COMPARISON
    if not args.only_omega and logs_trans is not None:
        print(f"\n{'='*100}")
        print(f"FINAL COMPARISON: Top Transformer  vs  OmegaGDM")
        print(f"{'='*100}")
        
        min_loss_t = min(l['loss'] for l in logs_trans)
        min_loss_o = min(l['loss'] for l in logs_omega)
        vram_t = max(l['vram_mb'] for l in logs_trans)
        vram_o = max(l['vram_mb'] for l in logs_omega)
        time_t = sum(l['step_time'] for l in logs_trans) / len(logs_trans)
        time_o = sum(l['step_time'] for l in logs_omega) / len(logs_omega)
        
        def diff_str(v_base, v_new, is_lower_better=True):
            if v_base == 0: return "N/A"
            pct = ((v_new - v_base) / v_base) * 100
            if is_lower_better:
                return f"{-pct:.1f}% better" if pct < 0 else f"{pct:.1f}% worse"
            return f"{pct:.1f}% larger" if pct > 0 else f"{-pct:.1f}% smaller"

        print(f"{'Metric':<28} | {'Top Transformer':<20} | {'OmegaGDM':<20} | {'OmegaGDM Improvement'}")
        print("-" * 100)
        print(f"{'Parameters':<28} | {params_trans/1e6:<18.2f} M | {params_omega/1e6:<18.2f} M | {diff_str(params_trans, params_omega, False)}")
        print(f"{'Minimum Loss':<28} | {min_loss_t:<20.4f} | {min_loss_o:<20.4f} | {diff_str(min_loss_t, min_loss_o)}")
        print(f"{'Final BPB':<28} | {logs_trans[-1]['bpb']:<20.4f} | {logs_omega[-1]['bpb']:<20.4f} | {diff_str(logs_trans[-1]['bpb'], logs_omega[-1]['bpb'])}")
        print(f"{'Peak Net VRAM':<28} | {vram_t:<18}MB | {vram_o:<18}MB | {diff_str(vram_t, vram_o)}")
        print(f"{'Avg Step Time':<28} | {time_t:<19.3f}s | {time_o:<19.3f}s | {diff_str(time_t, time_o)}")
        print(f"{'='*100}")
    else:
        print(f"\n{'='*100}")
        print(f"TRAINING COMPLETE: OmegaGDM V2.1 (New)")
        print(f"{'='*100}")
        min_loss_o = min(l['loss'] for l in logs_omega)
        vram_o = max(l['vram_mb'] for l in logs_omega)
        time_o = sum(l['step_time'] for l in logs_omega) / len(logs_omega)
        print(f"Parameters: {params_omega/1e6:.2f} M")
        print(f"Minimum Loss: {min_loss_o:.4f}")
        print(f"Final BPB: {logs_omega[-1]['bpb']:.4f}")
        print(f"Peak Net VRAM: {vram_o} MB")
        print(f"Avg Step Time: {time_o:.3f} s")
        print(f"{'='*100}")

    if args.quick:
        # Run interpret_omega.py to verify internals and print interpretability data
        print("\n" + "="*60)
        print("RUNNING INTERPRETABILITY PROBE")
        print("="*60)
        cmd = [sys.executable, "interpret_omega.py", "--model", "omega", "--ckpt", ckpt_path, "--out", "probe_data.json"]
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
