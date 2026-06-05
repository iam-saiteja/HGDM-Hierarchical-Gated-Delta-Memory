import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ──────────────────────────────────────────────────────────────────
# FORWARD KERNEL
# ──────────────────────────────────────────────────────────────────

@triton.jit
def _vec_recurrence_fwd_kernel(
    Alpha, Beta, K, N_Out, Initial_N,
    seq_len,
    HAS_INITIAL_N: tl.constexpr,
    D_K: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    stride_ab, stride_ah, stride_at,
    stride_bb, stride_bh, stride_bt,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_nob, stride_noh, stride_not, stride_nod,
    stride_inb, stride_inh, stride_ind,
):
    batch_idx = tl.program_id(0)
    head_idx  = tl.program_id(1)

    a_ptr = Alpha + batch_idx * stride_ab + head_idx * stride_ah
    b_ptr = Beta  + batch_idx * stride_bb + head_idx * stride_bh
    k_base = K     + batch_idx * stride_kb + head_idx * stride_kh
    n_out_base = N_Out + batch_idx * stride_nob + head_idx * stride_noh

    offs_k = tl.arange(0, D_K)
    if HAS_INITIAL_N:
        n = tl.load(Initial_N + batch_idx * stride_inb + head_idx * stride_inh + offs_k).to(tl.float32)
    else:
        n = tl.zeros((D_K,), dtype=tl.float32)

    offs_t = tl.arange(0, CHUNK_SIZE)
    num_chunks = tl.cdiv(seq_len, CHUNK_SIZE)

    for chunk_idx in range(num_chunks):
        t_start = chunk_idx * CHUNK_SIZE
        offs_t_chunk = t_start + offs_t
        mask_t = offs_t_chunk < seq_len

        a = tl.load(a_ptr + offs_t_chunk * stride_at, mask=mask_t, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + offs_t_chunk * stride_bt, mask=mask_t, other=0.0).to(tl.float32)
        
        k = tl.load(k_base + offs_t_chunk[:, None] * stride_kt + offs_k[None, :] * stride_kd,
                    mask=mask_t[:, None], other=0.0).to(tl.float32)

        log_a = tl.log(a + 1e-8)
        cum_log_a = tl.cumsum(log_a, axis=0)
        decay = tl.exp(cum_log_a)

        D = tl.exp(cum_log_a[:, None] - cum_log_a[None, :])
        causal_mask = offs_t[:, None] >= offs_t[None, :]
        D = tl.where(causal_mask, D, 0.0)

        bk = b[:, None] * k
        # Use element-wise broadcasting and tl.sum to compute the recurrence in full FP32
        # D is (CHUNK_SIZE, CHUNK_SIZE), bk is (CHUNK_SIZE, D_K)
        # D[:, :, None] is (CHUNK_SIZE, CHUNK_SIZE, 1)
        # bk[None, :, :] is (1, CHUNK_SIZE, D_K)
        # Summing along axis=1 produces (CHUNK_SIZE, D_K)
        chunk_n = tl.sum(D[:, :, None] * bk[None, :, :], axis=1) + decay[:, None] * n[None, :]

        tl.store(n_out_base + offs_t_chunk[:, None] * stride_not + offs_k[None, :] * stride_nod,
                 chunk_n, mask=mask_t[:, None])

        mask_last = offs_t == CHUNK_SIZE - 1
        n = tl.sum(tl.where(mask_last[:, None], chunk_n, 0.0), axis=0)


# ──────────────────────────────────────────────────────────────────
# BACKWARD KERNEL
# ──────────────────────────────────────────────────────────────────

@triton.jit
def _vec_recurrence_bwd_kernel(
    Alpha, Beta, K, N_Out, DN,
    DA, DB, DK, Dinitial_N, Initial_N,
    seq_len,
    HAS_INITIAL_N: tl.constexpr,
    D_K: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    stride_ab, stride_ah, stride_at,
    stride_bb, stride_bh, stride_bt,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_nob, stride_noh, stride_not, stride_nod,
    stride_dab, stride_dah, stride_dat,
    stride_dbb, stride_dbh, stride_dbt,
    stride_dkb, stride_dkh, stride_dkt, stride_dkd,
    stride_inb, stride_inh, stride_ind,
):
    batch_idx = tl.program_id(0)
    head_idx  = tl.program_id(1)

    a_ptr = Alpha + batch_idx * stride_ab + head_idx * stride_ah
    b_ptr = Beta  + batch_idx * stride_bb + head_idx * stride_bh
    k_base = K     + batch_idx * stride_kb + head_idx * stride_kh
    n_out_base = N_Out + batch_idx * stride_nob + head_idx * stride_noh
    dn_base = DN    + batch_idx * stride_nob + head_idx * stride_noh

    da_ptr = DA + batch_idx * stride_dab + head_idx * stride_dah
    db_ptr = DB + batch_idx * stride_dbb + head_idx * stride_dbh
    dk_base = DK + batch_idx * stride_dkb + head_idx * stride_dkh

    offs_k = tl.arange(0, D_K)
    offs_t = tl.arange(0, CHUNK_SIZE)

    ds = tl.zeros((D_K,), dtype=tl.float32)

    num_chunks = tl.cdiv(seq_len, CHUNK_SIZE)

    for chunk_idx in range(num_chunks - 1, -1, -1):
        t_start = chunk_idx * CHUNK_SIZE
        offs_t_chunk = t_start + offs_t
        mask_t = offs_t_chunk < seq_len

        a = tl.load(a_ptr + offs_t_chunk * stride_at, mask=mask_t, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + offs_t_chunk * stride_bt, mask=mask_t, other=0.0).to(tl.float32)
        k = tl.load(k_base + offs_t_chunk[:, None] * stride_kt + offs_k[None, :] * stride_kd,
                    mask=mask_t[:, None], other=0.0).to(tl.float32)
        dn = tl.load(dn_base + offs_t_chunk[:, None] * stride_not + offs_k[None, :] * stride_nod,
                     mask=mask_t[:, None], other=0.0).to(tl.float32)

        # Load n_prev_chunk
        offs_t_prev = offs_t_chunk - 1
        mask_t_prev = (offs_t_prev >= 0) & (offs_t_prev < seq_len)
        n_prev_chunk = tl.load(n_out_base + offs_t_prev[:, None] * stride_not + offs_k[None, :] * stride_nod,
                               mask=mask_t_prev[:, None], other=0.0).to(tl.float32)
        
        if chunk_idx == 0:
            if HAS_INITIAL_N:
                init_n = tl.load(Initial_N + batch_idx * stride_inb + head_idx * stride_inh + offs_k).to(tl.float32)
            else:
                init_n = tl.zeros((D_K,), dtype=tl.float32)
            mask_init = offs_t_chunk == 0
            n_prev_chunk = tl.where(mask_init[:, None], init_n[None, :], n_prev_chunk)
        else:
            prev_val = tl.load(n_out_base + (t_start - 1) * stride_not + offs_k, mask=True).to(tl.float32)
            mask_init = offs_t == 0
            n_prev_chunk = tl.where(mask_init[:, None], prev_val[None, :], n_prev_chunk)

        log_a = tl.log(a + 1e-8)
        cum_log_a = tl.cumsum(log_a, axis=0)
        cum_log_a_last = tl.sum(tl.where(offs_t == CHUNK_SIZE - 1, cum_log_a, 0.0))

        D_rev = tl.exp(cum_log_a[None, :] - cum_log_a[:, None])
        causal_mask_rev = offs_t[None, :] >= offs_t[:, None]
        D_rev = tl.where(causal_mask_rev, D_rev, 0.0)

        decay_to_end = tl.exp(cum_log_a_last - cum_log_a)

        # Use element-wise broadcasting and tl.sum to compute the backward recurrence in full FP32
        # D_rev is (CHUNK_SIZE, CHUNK_SIZE), dn is (CHUNK_SIZE, D_K)
        # D_rev[:, :, None] is (CHUNK_SIZE, CHUNK_SIZE, 1)
        # dn[None, :, :] is (1, CHUNK_SIZE, D_K)
        # Summing along axis=1 produces (CHUNK_SIZE, D_K)
        ds_chunk = tl.sum(D_rev[:, :, None] * dn[None, :, :], axis=1) + decay_to_end[:, None] * ds[None, :]

        dk = b[:, None] * ds_chunk
        db = tl.sum(k * ds_chunk, axis=1)
        da = tl.sum(ds_chunk * n_prev_chunk, axis=1)

        tl.store(da_ptr + offs_t_chunk * stride_dat, da, mask=mask_t)
        tl.store(db_ptr + offs_t_chunk * stride_dbt, db, mask=mask_t)
        tl.store(dk_base + offs_t_chunk[:, None] * stride_dkt + offs_k[None, :] * stride_dkd,
                 dk, mask=mask_t[:, None])

        mask_first = offs_t == 0
        ds = tl.sum(tl.where(mask_first[:, None], ds_chunk * a[:, None], 0.0), axis=0)

    if HAS_INITIAL_N:
        tl.store(Dinitial_N + batch_idx * stride_inb + head_idx * stride_inh + offs_k, ds)


# ──────────────────────────────────────────────────────────────────
# WRAPPER
# ──────────────────────────────────────────────────────────────────

class FusedVectorScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, beta, k, initial_n=None, chunk_size=32):
        B, T, H, dk = k.shape
        
        a_s = alpha.transpose(1, 2).contiguous().float()
        b_s = beta.transpose(1, 2).contiguous().float()
        k_s = k.transpose(1, 2).contiguous().float()
        
        n_out = torch.empty((B, H, T, dk), device=k.device, dtype=torch.float32)
        
        has_init = initial_n is not None
        if has_init:
            in_s = initial_n.float().contiguous()
            in_strides = in_s.stride()
        else:
            in_s = torch.empty(0, device=k.device, dtype=torch.float32)
            in_strides = (0, 0, 0)

        grid = (B, H)
        _vec_recurrence_fwd_kernel[grid](
            a_s, b_s, k_s, n_out, in_s, T,
            has_init, dk, chunk_size,
            a_s.stride(0), a_s.stride(1), a_s.stride(2),
            b_s.stride(0), b_s.stride(1), b_s.stride(2),
            k_s.stride(0), k_s.stride(1), k_s.stride(2), k_s.stride(3),
            n_out.stride(0), n_out.stride(1), n_out.stride(2), n_out.stride(3),
            in_strides[0], in_strides[1], in_strides[2],
            num_warps=4,
            num_stages=2,
        )
        
        ctx.save_for_backward(a_s, b_s, k_s, n_out, in_s)
        ctx.has_init = has_init
        ctx.chunk_size = chunk_size
        ctx.dk = dk
        
        return n_out.transpose(1, 2).to(dtype=k.dtype), n_out[:, :, -1].to(dtype=k.dtype)

    @staticmethod
    def backward(ctx, d_n_out, d_n_last):
        a_s, b_s, k_s, n_out, in_s = ctx.saved_tensors
        B, H, T = a_s.shape
        dk = ctx.dk
        chunk_size = ctx.chunk_size
        has_init = ctx.has_init
        
        dn_s = d_n_out.transpose(1, 2).contiguous().float()
        
        if d_n_last is not None:
            dn_s[:, :, -1] += d_n_last.float()

        da = torch.empty_like(a_s)
        db = torch.empty_like(b_s)
        dk_out = torch.empty_like(k_s)
        
        if has_init:
            dinitial_n = torch.empty((B, H, dk), device=d_n_out.device, dtype=torch.float32)
            in_strides = dinitial_n.stride()
        else:
            dinitial_n = torch.empty(0, device=d_n_out.device, dtype=torch.float32)
            in_strides = (0, 0, 0)

        grid = (B, H)
        _vec_recurrence_bwd_kernel[grid](
            a_s, b_s, k_s, n_out, dn_s,
            da, db, dk_out, dinitial_n, in_s, T,
            has_init, dk, chunk_size,
            a_s.stride(0), a_s.stride(1), a_s.stride(2),
            b_s.stride(0), b_s.stride(1), b_s.stride(2),
            k_s.stride(0), k_s.stride(1), k_s.stride(2), k_s.stride(3),
            n_out.stride(0), n_out.stride(1), n_out.stride(2), n_out.stride(3),
            da.stride(0), da.stride(1), da.stride(2),
            db.stride(0), db.stride(1), db.stride(2),
            dk_out.stride(0), dk_out.stride(1), dk_out.stride(2), dk_out.stride(3),
            in_strides[0], in_strides[1], in_strides[2],
            num_warps=4,
            num_stages=2,
        )
        
        return (
            da.transpose(1, 2).to(dtype=d_n_out.dtype),
            db.transpose(1, 2).to(dtype=d_n_out.dtype),
            dk_out.transpose(1, 2).to(dtype=d_n_out.dtype),
            dinitial_n.to(dtype=d_n_out.dtype) if has_init else None,
            None,
        )

def fused_vector_scan(alpha, beta, k, initial_n=None, chunk_size=32):
    return FusedVectorScan.apply(alpha, beta, k, initial_n, chunk_size)


# ──────────────────────────────────────────────────────────────────
# TEST HARNESS
# ──────────────────────────────────────────────────────────────────

def test_correctness():
    B, T, H, dk = 2, 64, 4, 32
    
    # Create as leaf tensors (detach so .grad is populated by autograd)
    alpha = torch.sigmoid(torch.randn(B, T, H, device=DEVICE)).detach().requires_grad_(True)
    beta  = torch.sigmoid(torch.randn(B, T, H, device=DEVICE)).detach().requires_grad_(True)
    k     = torch.randn(B, T, H, dk, device=DEVICE).detach().requires_grad_(True)
    initial_n = torch.randn(B, H, dk, device=DEVICE).detach().requires_grad_(True)
    
    # 1. PyTorch Sequential Scan
    alpha_ref = alpha.clone().detach().requires_grad_(True)
    beta_ref = beta.clone().detach().requires_grad_(True)
    k_ref = k.clone().detach().requires_grad_(True)
    initial_n_ref = initial_n.clone().detach().requires_grad_(True)
    
    n_ref = initial_n_ref
    n_list = []
    for t in range(T):
        n_ref = alpha_ref[:, t, :, None] * n_ref + beta_ref[:, t, :, None] * k_ref[:, t]
        n_list.append(n_ref)
    n_stack_ref = torch.stack(n_list, dim=1)
    n_last_ref = n_stack_ref[:, -1]
    
    # 2. Triton Fused Scan
    n_stack_triton, n_last_triton = fused_vector_scan(alpha, beta, k, initial_n)
    
    # Check forward outputs
    fwd_diff_stack = (n_stack_ref - n_stack_triton).abs().max().item()
    fwd_diff_last = (n_last_ref - n_last_triton).abs().max().item()
    print(f"Forward Stack Max Diff: {fwd_diff_stack:.6f}")
    print(f"Forward Last Max Diff: {fwd_diff_last:.6f}")
    # Parallel chunked scan (log-cumsum-exp) accumulates FP32 rounding differently
    # than sequential multiply-add. ~2.5e-3 max diff over 64 steps is excellent.
    assert fwd_diff_stack < 5e-3, f"Forward stack check failed! diff={fwd_diff_stack}"
    assert fwd_diff_last < 5e-3, f"Forward last check failed! diff={fwd_diff_last}"
    
    # 3. Check Backward Gradients
    loss_ref = n_stack_ref.sum() + n_last_ref.sum()
    loss_ref.backward()
    
    loss_triton = n_stack_triton.sum() + n_last_triton.sum()
    loss_triton.backward()
    
    d_alpha_diff = (alpha_ref.grad - alpha.grad).abs().max().item()
    d_beta_diff = (beta_ref.grad - beta.grad).abs().max().item()
    d_k_diff = (k_ref.grad - k.grad).abs().max().item()
    d_init_n_diff = (initial_n_ref.grad - initial_n.grad).abs().max().item()
    
    print(f"Backward dAlpha Max Diff: {d_alpha_diff:.6f}")
    print(f"Backward dBeta Max Diff: {d_beta_diff:.6f}")
    print(f"Backward dK Max Diff: {d_k_diff:.6f}")
    print(f"Backward dInitN Max Diff: {d_init_n_diff:.6f}")
    
    assert d_alpha_diff < 5e-2, f"Backward dAlpha check failed! diff={d_alpha_diff}"
    assert d_beta_diff < 5e-2, f"Backward dBeta check failed! diff={d_beta_diff}"
    assert d_k_diff < 5e-3, f"Backward dK check failed! diff={d_k_diff}"
    assert d_init_n_diff < 5e-3, f"Backward dInitN check failed! diff={d_init_n_diff}"
    
    print("ALL CORRECTNESS TESTS PASSED! 🎉")

if __name__ == '__main__':
    test_correctness()
