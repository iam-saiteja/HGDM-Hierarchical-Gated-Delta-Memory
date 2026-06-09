import torch
import torch.nn.functional as F
from hgdm_omega import OmegaGDM, OmegaConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = OmegaConfig(
    d_byte=256, catcher_layers=1, renderer_layers=1,
    d_model=256, core_layers=4, n_heads=4,
    d_k=32, d_v=32, d_ff=1024,
    decimation_rate=8, max_position_embeddings=512,
    vocab_size=256, use_state_fusion=False
)
model = OmegaGDM(cfg, force_sequential=False).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Generate dummy text data (representing text bytes)
data = torch.randint(0, 256, (10000,), device=device)

print("Starting mini-training test for 50 steps...")
for step in range(50):
    ix = torch.randint(0, len(data) - 512 - 1, (4,))
    x = torch.stack([data[i:i+512] for i in ix])
    y = torch.stack([data[i+1:i+512+1] for i in ix])
    
    opt.zero_grad()
    logits, _ = model(x)
    loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
    loss.backward()
    opt.step()
    
    if step % 10 == 0 or step == 49:
        print(f"Step {step:02d} | Loss: {loss.item():.4f}")
