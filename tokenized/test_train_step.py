import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tokenized.config import TokenizedConfig
from tokenized.model_hgdm import TokenizedHGDM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

cfg = TokenizedConfig.get_tier("pilot", vocab_size=50257)
cfg.use_hbs = True

print(f"Creating TokenizedHGDM with HBS (use_hbs={cfg.use_hbs})...")
model = TokenizedHGDM(cfg).to(device)

B, T = 4, 512
x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
y = torch.randint(0, cfg.vocab_size, (B, T), device=device)

print(f"Running forward pass with batch [{B}, {T}] in bfloat16...")
with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
    logits, _ = model(x)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))

print(f"Loss = {loss.item():.4f}")
print("Running backward pass...")
loss.backward()
print("SUCCESS: Backward pass completed cleanly with zero errors!")

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable Parameters: {param_count:,}")
vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
print(f"Peak VRAM: {vram_mb:.1f} MB")
