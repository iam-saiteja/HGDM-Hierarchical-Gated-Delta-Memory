import torch
import triton
import triton.language as tl
import time

@triton.jit
def _fused_step_fwd_kernel(
    Q, K, V, Alpha, Beta, OutGate,
    S_prev, S_out,
    N_prev, N_out,
    Out,
    USE_EPISTEMIC: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_ab, stride_ah,
    stride_bb, stride_bh,
    stride_gb, stride_gh, stride_gd,
    stride_sb, stride_sh, stride_sk, stride_sv,
    stride_sob, stride_soh, stride_sok, stride_sov,
    stride_nb, stride_nh, stride_nk,
    stride_nob, stride_noh, stride_nok,
    stride_ob, stride_oh, stride_od,
):
    batch_idx = tl.program_id(0)
    head_idx  = tl.program_id(1)

    offs_k = tl.arange(0, D_K)
    offs_v = tl.arange(0, D_V)

    # Base pointers
    q_ptr = Q + batch_idx * stride_qb + head_idx * stride_qh + offs_k * stride_qd
    k_ptr = K + batch_idx * stride_kb + head_idx * stride_kh + offs_k * stride_kd
    v_ptr = V + batch_idx * stride_vb + head_idx * stride_vh + offs_v * stride_vd
    a_ptr = Alpha + batch_idx * stride_ab + head_idx * stride_ah
    b_ptr = Beta  + batch_idx * stride_bb + head_idx * stride_bh
    g_ptr = OutGate + batch_idx * stride_gb + head_idx * stride_gh + offs_v * stride_gd

    s_ptr = S_prev + batch_idx * stride_sb + head_idx * stride_sh + offs_k[:, None] * stride_sk + offs_v[None, :] * stride_sv
    s_out_ptr = S_out + batch_idx * stride_sob + head_idx * stride_soh + offs_k[:, None] * stride_sok + offs_v[None, :] * stride_sov

    o_ptr = Out + batch_idx * stride_ob + head_idx * stride_oh + offs_v * stride_od

    # Load vectors & scalars
    alpha = tl.load(a_ptr).to(tl.float32)
    beta  = tl.load(b_ptr).to(tl.float32)
    q     = tl.load(q_ptr).to(tl.float32)
    k     = tl.load(k_ptr).to(tl.float32)
    v     = tl.load(v_ptr).to(tl.float32)
    gate  = tl.load(g_ptr).to(tl.float32)

    # Load S_prev [D_K, D_V]
    S = tl.load(s_ptr).to(tl.float32)

    # 1. State update: S_new = alpha * S + beta * (k * v)
    delta = k[:, None] * v[None, :]
    S_new = alpha * S + beta * delta
    tl.store(s_out_ptr, S_new)

    # 2. Query readout: out_v = sum_k (S_new[k, v] * q[k])
    out_v = tl.sum(S_new * q[:, None], axis=0)

    # 3. Epistemic Normalization
    if USE_EPISTEMIC:
        n_ptr = N_prev + batch_idx * stride_nb + head_idx * stride_nh + offs_k * stride_nk
        n_out_ptr = N_out + batch_idx * stride_nob + head_idx * stride_noh + offs_k * stride_nok
        
        n_prev = tl.load(n_ptr).to(tl.float32)
        n_new = alpha * n_prev + beta * k
        tl.store(n_out_ptr, n_new)

        n_abs = tl.abs(n_new)
        n_inf = tl.max(n_abs, axis=0)
        denom = tl.maximum(n_inf, 1.0)

        n_sq = n_new * n_new
        norm_n = tl.sqrt(tl.sum(n_sq, axis=0) + 1e-8)
        conf = 2.0 * tl.sigmoid(2.0 * norm_n) - 1.0

        out_v = (out_v / denom) * conf

    out_final = out_v * gate
    tl.store(o_ptr, out_final)


def fused_step_triton(q, k, v, alpha, beta, out_gate, S_prev, n_prev=None):
    """
    Triton wrapper for single-token autoregressive step.
    Inputs:
      q, k: [B, H, dk]
      v, out_gate: [B, H, dv]
      alpha, beta: [B, H]
      S_prev: [B, H, dk, dv]
      n_prev: [B, H, dk] (optional)
    """
    B, H, dk = q.shape
    dv = v.shape[-1]
    
    out = torch.empty((B, H, dv), device=q.device, dtype=q.dtype)
    S_out = torch.empty_like(S_prev)
    use_epistemic = n_prev is not None
    if use_epistemic:
        n_out = torch.empty_like(n_prev)
    else:
        n_out = torch.empty(0, device=q.device, dtype=q.dtype)
        n_prev = torch.empty(0, device=q.device, dtype=q.dtype)

    grid = (B, H)
    _fused_step_fwd_kernel[grid](
        q, k, v, alpha, beta, out_gate,
        S_prev, S_out,
        n_prev, n_out,
        out,
        USE_EPISTEMIC=use_epistemic,
        D_K=dk, D_V=dv,
        stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qd=q.stride(2),
        stride_kb=k.stride(0), stride_kh=k.stride(1), stride_kd=k.stride(2),
        stride_vb=v.stride(0), stride_vh=v.stride(1), stride_vd=v.stride(2),
        stride_ab=alpha.stride(0), stride_ah=alpha.stride(1),
        stride_bb=beta.stride(0), stride_bh=beta.stride(1),
        stride_gb=out_gate.stride(0), stride_gh=out_gate.stride(1), stride_gd=out_gate.stride(2),
        stride_sb=S_prev.stride(0), stride_sh=S_prev.stride(1), stride_sk=S_prev.stride(2), stride_sv=S_prev.stride(3),
        stride_sob=S_out.stride(0), stride_soh=S_out.stride(1), stride_sok=S_out.stride(2), stride_sov=S_out.stride(3),
        stride_nb=n_prev.stride(0) if use_epistemic else 0, stride_nh=n_prev.stride(1) if use_epistemic else 0, stride_nk=n_prev.stride(2) if use_epistemic else 0,
        stride_nob=n_out.stride(0) if use_epistemic else 0, stride_noh=n_out.stride(1) if use_epistemic else 0, stride_nok=n_out.stride(2) if use_epistemic else 0,
        stride_ob=out.stride(0), stride_oh=out.stride(1), stride_od=out.stride(2),
        num_warps=2,
    )
    return out, S_out, (n_out if use_epistemic else None)


def pytorch_step_reference(q, k, v, alpha, beta, out_gate, S_prev, n_prev=None):
    # PyTorch reference
    delta = torch.einsum('bhk,bhd->bhkd', k, v)
    S_new = alpha[:, :, None, None] * S_prev + beta[:, :, None, None] * delta
    out_t = torch.einsum('bhkd,bhk->bhd', S_new, q)
    if n_prev is not None:
        n_new = alpha[:, :, None] * n_prev + beta[:, :, None] * k
        n_inf = n_new.abs().max(dim=-1)[0]
        denom = torch.clamp(n_inf, min=1.0).unsqueeze(-1)
        conf = torch.tanh(n_new.norm(dim=-1)).unsqueeze(-1)
        out_t = (out_t / denom) * conf
    else:
        n_new = None
    out_t = out_t * out_gate
    return out_t, S_new, n_new


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
        
    device = "cuda"
    B, H, dk, dv = 1, 6, 64, 64
    dtype = torch.bfloat16
    
    q = torch.randn(B, H, dk, device=device, dtype=dtype)
    k = torch.randn(B, H, dk, device=device, dtype=dtype)
    v = torch.randn(B, H, dv, device=device, dtype=dtype)
    alpha = torch.rand(B, H, device=device, dtype=dtype)
    beta = torch.rand(B, H, device=device, dtype=dtype)
    out_gate = torch.randn(B, H, dv, device=device, dtype=dtype)
    S_prev = torch.randn(B, H, dk, dv, device=device, dtype=dtype)
    n_prev = torch.randn(B, H, dk, device=device, dtype=dtype)

    # 1. Numerical verification
    out_ref, S_ref, n_ref = pytorch_step_reference(q, k, v, alpha, beta, out_gate, S_prev, n_prev)
    out_tri, S_tri, n_tri = fused_step_triton(q, k, v, alpha, beta, out_gate, S_prev, n_prev)

    diff_out = (out_ref - out_tri).abs().max().item()
    diff_S = (S_ref - S_tri).abs().max().item()
    diff_n = (n_ref - n_tri).abs().max().item()

    print(f"[Verification] Max diff Out: {diff_out:.6f}")
    print(f"[Verification] Max diff S:   {diff_S:.6f}")
    print(f"[Verification] Max diff n:   {diff_n:.6f}")

    # 2. Benchmark PyTorch loop vs Triton step (1000 steps)
    N_STEPS = 1000
    torch.cuda.synchronize()
    t0 = time.time()
    S_cur = S_prev.clone()
    n_cur = n_prev.clone()
    for _ in range(N_STEPS):
        _, S_cur, n_cur = pytorch_step_reference(q, k, v, alpha, beta, out_gate, S_cur, n_cur)
    torch.cuda.synchronize()
    t_pytorch = time.time() - t0
    speed_pytorch = N_STEPS / t_pytorch

    torch.cuda.synchronize()
    t0 = time.time()
    S_cur = S_prev.clone()
    n_cur = n_prev.clone()
    for _ in range(N_STEPS):
        _, S_cur, n_cur = fused_step_triton(q, k, v, alpha, beta, out_gate, S_cur, n_cur)
    torch.cuda.synchronize()
    t_triton = time.time() - t0
    speed_triton = N_STEPS / t_triton

    print(f"\n[Speed Comparison - 1 Layer, 1000 Steps]:")
    print(f"  PyTorch: {speed_pytorch:.1f} steps/s ({t_pytorch*1000:.2f} ms total)")
    print(f"  Triton:  {speed_triton:.1f} steps/s ({t_triton*1000:.2f} ms total)")
    print(f"  Speedup: {speed_triton / speed_pytorch:.2f}x faster!")

if __name__ == "__main__":
    main()
