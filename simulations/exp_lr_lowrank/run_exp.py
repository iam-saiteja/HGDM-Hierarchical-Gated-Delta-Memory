import argparse
import json
import time
import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model_lowrank import HGDMLowRankUltimate
from hgdm_ultimate import HGDMConfig

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=200)
parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--batch', type=int, default=2)
parser.add_argument('--r', type=int, default=8)
parser.add_argument('--small-run', action='store_true')
args = parser.parse_args()

# Minimal synthetic run for quick verification
config = HGDMConfig(d_model=256, n_layers=2, n_heads=4, d_k=64, d_v=64, d_ff=512, vocab_size=256)
model = HGDMLowRankUltimate(config, n_layers=2, r=args.r)

device = torch.device('cuda')
model.to(device)

# synthetic data: random bytes with a next-token shift task
B = args.batch
T = args.seq_len
train = torch.randint(0, 256, (B, T), dtype=torch.long, device=device)
target = torch.roll(train, shifts=-1, dims=1)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

history = []
start = time.time()
for step in range(args.steps):
    opt.zero_grad()
    logits, _ = model(train)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 256), target.view(-1))
    loss.backward()
    opt.step()
    if step % 50 == 0:
        elapsed = time.time() - start
        print(f"Step {step} | Loss {loss.item():.4f} | Elapsed {elapsed:.1f}s")
        history.append((step, loss.item()))

res = {
    'final_loss': float(loss.item()),
    'steps': args.steps,
    'r': args.r,
    'history': history
}
with open('results.json', 'w') as f:
    json.dump(res, f, indent=2)

print('Done. results.json written.')
