import torch
from ultimate.hgdm_ultimate import HGDMUltimate, HGDMConfig

config = HGDMConfig(use_rope=True, use_epistemic_gate=True, n_grad_mode='exact')
model = HGDMUltimate(config).cuda()
x = torch.randint(0, 256, (2, 64)).cuda()
out, _ = model(x)
loss = out.sum()
loss.backward()
print("Pass")
