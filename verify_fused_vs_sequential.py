import torch
from hgdm_ultimate import MultiHeadGatedDelta, HGDMConfig
from kernel_nitro import fused_nitro_scan

def sequential_raw_demo(model, z):
    B, T, _ = z.shape
    m = model
    q = m.W_q(z).view(B, T, m.H, m.d_k)
    k = m.W_k(z).view(B, T, m.H, m.d_k)
    v = m.W_v(z).view(B, T, m.H, m.d_v)
    alpha = torch.sigmoid(m.W_alpha(z))
    beta = torch.sigmoid(m.W_beta(z))

    S = torch.zeros(B, m.H, m.d_k, m.d_v, device=z.device, dtype=z.dtype)
    outs = []
    for t in range(T):
        delta = torch.einsum('bhk,bhd->bhkd', k[:, t], v[:, t])
        S = alpha[:, t, :, None, None] * S + beta[:, t, :, None, None] * delta
        out_t = torch.einsum('bhkd,bhk->bhd', S, q[:, t])
        outs.append(out_t)
    return torch.stack(outs, dim=1), S


def fused_demo(model, z):
    B, T, C = z.shape
    m = model
    q = m.W_q(z).view(B, T, m.H, m.d_k)
    k = m.W_k(z).view(B, T, m.H, m.d_k)
    v = m.W_v(z).view(B, T, m.H, m.d_v)
    alpha = torch.sigmoid(m.W_alpha(z))
    beta = torch.sigmoid(m.W_beta(z))
    if fused_nitro_scan is None:
        print('fused_nitro_scan not available in this environment.')
        return None, None
    out, S = fused_nitro_scan(q, k, v, alpha, beta, None)
    return out, S

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    config = HGDMConfig(d_model=128, n_layers=1, n_heads=4, d_k=64, d_v=64)
    model = MultiHeadGatedDelta(config, force_sequential=True).to(device)
    model.eval()
    z = torch.randn(2, 16, 128, device=device)
    out_seq, S_seq = sequential_raw_demo(model, z)
    fused = fused_demo(model, z)
    if fused[0] is None:
        print('Skipping fused vs sequential check; fused kernel unavailable.')
    else:
        out_fused, S_fused = fused
        diff = (out_seq - out_fused).abs().max().item()
        state_diff = (S_seq - S_fused).abs().max().item()
        print('Max abs diff between sequential and fused raw outputs:', diff)
        print('Max abs diff between sequential and fused states:', state_diff)
