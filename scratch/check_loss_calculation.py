import torch
import torch.nn.functional as F
from hgdm_omega import OmegaGDM, OmegaConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = OmegaConfig(
    d_byte=256, catcher_layers=1, renderer_layers=1,
    d_model=512, core_layers=8, n_heads=8,
    d_k=64, d_v=64, d_ff=2048,
    decimation_rate=8, max_position_embeddings=2048,
    vocab_size=256, use_state_fusion=False
)
model = OmegaGDM(cfg, force_sequential=False).to(device)
x = torch.randint(0, 256, (8, 2048), device=device)
y = torch.randint(0, 256, (8, 2048), device=device)

logits, _ = model(x)
print("Logits stats:")
print("  min:", logits.min().item())
print("  max:", logits.max().item())
print("  mean:", logits.mean().item())
print("  std:", logits.std().item())

loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
print("loss:", loss.item())
