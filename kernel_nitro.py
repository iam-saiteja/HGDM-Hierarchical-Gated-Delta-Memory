"""
kernel_nitro.py — Fixed Triton Kernel for HGDM
Corrections from v6:
  1. Removed Python None-check inside @triton.jit (was causing static compilation bug)
  2. Fixed num_warps=4 (was 2 — left half of Ampere cores idle)
  3. Fixed M = D * b[:, None] (was b[None,:] — wrong axis, silent math error)
  4. num_stages=2 for better memory prefetching on Ampere
Corrections from v7:
  5. Decoupled K strides from Q strides — forward and backward kernels now accept
     independent stride_kb/kh/kt/kd arguments for K, enabling correct operation
     under GQA/MQA layouts where Q and K have different head counts or d_k.
Corrections from v8:
  6. Parameterised block dimensions — D_K, D_V, and CHUNK_SIZE are now tl.constexpr
     arguments. All previously hardcoded 64/32/31 literals are replaced with these
     symbols, allowing the kernel to be compiled at any valid power-of-2 size
     (e.g. D_K=32, D_V=32, CHUNK_SIZE=16) without modifying kernel source.
Corrections from v9:
  7. Cross-segment recurrent gradient backpropagation — backward pass now accepts
     dstate (gradient with respect to final state S) and propagates it back to
     dinitial_state (gradient with respect to initial state S_prev).
"""
import torch
import triton
import triton.language as tl
import math


# ──────────────────────────────────────────────────────────────────
# FORWARD KERNEL
# ──────────────────────────────────────────────────────────────────

@triton.jit
def _chunk_fwd_kernel(
    Q, K, V, Alpha, Beta,
    Out, State, Initial_State,
    seq_len,
    HAS_INITIAL_STATE: tl.constexpr,   # FIX 1: constexpr, not runtime None check
    D_K: tl.constexpr,                 # FIX 6: parameterised key dimension
    D_V: tl.constexpr,                 # FIX 6: parameterised value dimension
    CHUNK_SIZE: tl.constexpr,          # FIX 6: parameterised chunk length
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,  # FIX 5: independent K strides
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ab, stride_ah, stride_at,
    stride_bb, stride_bh, stride_bt,
    stride_sb, stride_sh, stride_sc, stride_sk, stride_sv,
    stride_isb, stride_ish, stride_isk, stride_isv,
):
    batch_idx = tl.program_id(0)
    head_idx  = tl.program_id(1)

    q_base = Q     + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K     + batch_idx * stride_kb + head_idx * stride_kh  # FIX 5: use K strides
    v_base = V     + batch_idx * stride_vb + head_idx * stride_vh
    a_base = Alpha + batch_idx * stride_ab + head_idx * stride_ah
    b_base = Beta  + batch_idx * stride_bb + head_idx * stride_bh
    o_base = Out   + batch_idx * stride_vb + head_idx * stride_vh

    offs_k = tl.arange(0, D_K)       # FIX 6
    offs_v = tl.arange(0, D_V)       # FIX 6
    offs_t = tl.arange(0, CHUNK_SIZE) # FIX 6

    # FIX 1: constexpr branch — compiles to two separate PTX variants
    if HAS_INITIAL_STATE:
        is_ptr = Initial_State + batch_idx * stride_isb + head_idx * stride_ish
        S = tl.load(
            is_ptr + offs_k[:, None] * stride_isk + offs_v[None, :] * stride_isv
        ).to(tl.float32)
    else:
        S = tl.zeros((D_K, D_V), dtype=tl.float32)  # FIX 6

    num_chunks = tl.cdiv(seq_len, CHUNK_SIZE)  # FIX 6

    for chunk_idx in range(num_chunks):
        t_start      = chunk_idx * CHUNK_SIZE  # FIX 6
        offs_t_chunk = t_start + offs_t
        mask_t       = offs_t_chunk < seq_len

        q = tl.load(q_base + offs_t_chunk[:, None] * stride_qt + offs_k[None, :] * stride_qd,
                    mask=mask_t[:, None], other=0.0).to(tl.float32)
        k = tl.load(k_base + offs_t_chunk[:, None] * stride_kt + offs_k[None, :] * stride_kd,  # FIX 5
                    mask=mask_t[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_base + offs_t_chunk[:, None] * stride_vt + offs_v[None, :] * stride_vd,
                    mask=mask_t[:, None], other=0.0).to(tl.float32)
        a = tl.load(a_base + offs_t_chunk * stride_at, mask=mask_t, other=1.0).to(tl.float32)
        b = tl.load(b_base + offs_t_chunk * stride_bt, mask=mask_t, other=0.0).to(tl.float32)

        log_a     = tl.log(a + 1e-8)
        cum_log_a = tl.cumsum(log_a, axis=0)          # (CHUNK_SIZE,)

        # Intra-chunk decay matrix: D[i,j] = exp(cum[i] - cum[j]) for i >= j
        D            = tl.exp(cum_log_a[:, None] - cum_log_a[None, :])
        causal_mask  = offs_t[:, None] >= offs_t[None, :]
        D            = tl.where(causal_mask, D, 0.0)

        # beta applied at write time (columns, not rows)
        # M[i,j] = D[i,j] * b[j] — position j is the write position
        M = D * b[None, :]

        QK        = tl.dot(q, tl.trans(k))
        out_intra = tl.dot(QK * M, v)

        decay     = tl.exp(cum_log_a)
        out_inter = tl.dot(q, S) * decay[:, None]

        out = out_intra + out_inter
        tl.store(o_base + offs_t_chunk[:, None] * stride_vt + offs_v[None, :] * stride_vd,
                 out, mask=mask_t[:, None])

        # State update — carry forward to next chunk
        mask_last      = offs_t == CHUNK_SIZE - 1  # FIX 6: was hardcoded 31
        cum_log_a_last = tl.sum(tl.where(mask_last, cum_log_a, 0.0))
        last_decay     = tl.exp(cum_log_a_last)

        # FIX 3: D_last_row[j] = exp(cum_last - cum[j]) — decay from j to end of chunk
        # coeff[j] = D_last_row[j] * b[j]  — b at write position j
        D_last_row = tl.exp(cum_log_a_last - cum_log_a)
        coeff      = D_last_row * b

        k_weighted   = k * coeff[:, None]
        chunk_update = tl.dot(tl.trans(k_weighted), v)
        S            = S * last_decay + chunk_update

        # Save state for backward pass
        state_ptr = State + batch_idx * stride_sb + head_idx * stride_sh + chunk_idx * stride_sc
        tl.store(
            state_ptr + offs_k[:, None] * stride_sk + offs_v[None, :] * stride_sv,
            S,
            mask=(offs_k[:, None] < D_K) & (offs_v[None, :] < D_V)  # FIX 6
        )


# ──────────────────────────────────────────────────────────────────
# BACKWARD KERNEL
# ──────────────────────────────────────────────────────────────────

@triton.jit
def _chunk_bwd_kernel(
    Q, K, V, Alpha, Beta, State, Dout,
    DQ, DK, DV, DAlpha, DBeta,
    Dstate, Dinitial_state,            # FIX 7: Cross-segment recurrent state gradients
    seq_len,
    D_K: tl.constexpr,                 # FIX 6
    D_V: tl.constexpr,                 # FIX 6
    CHUNK_SIZE: tl.constexpr,          # FIX 6
    HAS_DSTATE: tl.constexpr,         # FIX 7: dstate presence
    HAS_DINITIAL_STATE: tl.constexpr, # FIX 7: dinitial_state presence
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,  # FIX 5: independent K strides
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ab, stride_ah, stride_at,
    stride_bb, stride_bh, stride_bt,
    stride_sb, stride_sh, stride_sc, stride_sk, stride_sv,
    stride_dsb, stride_dsh, stride_dsk, stride_dsv,       # FIX 7
    stride_disb, stride_dish, stride_disk, stride_disv,   # FIX 7
):
    batch_idx = tl.program_id(0)
    head_idx  = tl.program_id(1)

    q_base    = Q     + batch_idx * stride_qb + head_idx * stride_qh
    k_base    = K     + batch_idx * stride_kb + head_idx * stride_kh  # FIX 5
    v_base    = V     + batch_idx * stride_vb + head_idx * stride_vh
    a_base    = Alpha + batch_idx * stride_ab + head_idx * stride_ah
    b_base    = Beta  + batch_idx * stride_bb + head_idx * stride_bh
    dout_base = Dout  + batch_idx * stride_vb + head_idx * stride_vh
    dq_base   = DQ    + batch_idx * stride_qb + head_idx * stride_qh
    dk_base   = DK    + batch_idx * stride_kb + head_idx * stride_kh  # FIX 5
    dv_base   = DV    + batch_idx * stride_vb + head_idx * stride_vh
    da_base   = DAlpha + batch_idx * stride_ab + head_idx * stride_ah
    db_base   = DBeta  + batch_idx * stride_bb + head_idx * stride_bh

    offs_k = tl.arange(0, D_K)        # FIX 6
    offs_v = tl.arange(0, D_V)        # FIX 6
    offs_t = tl.arange(0, CHUNK_SIZE) # FIX 6

    # FIX 7: Initialize dS from dstate (if provided), otherwise from zeros
    if HAS_DSTATE:
        dstate_ptr = Dstate + batch_idx * stride_dsb + head_idx * stride_dsh
        dS = tl.load(
            dstate_ptr + offs_k[:, None] * stride_dsk + offs_v[None, :] * stride_dsv
        ).to(tl.float32)
    else:
        dS = tl.zeros((D_K, D_V), dtype=tl.float32)  # FIX 6

    num_chunks = tl.cdiv(seq_len, CHUNK_SIZE)              # FIX 6

    for chunk_idx in range(num_chunks - 1, -1, -1):
        t_start      = chunk_idx * CHUNK_SIZE  # FIX 6
        offs_t_chunk = t_start + offs_t
        mask_t       = offs_t_chunk < seq_len

        q    = tl.load(q_base    + offs_t_chunk[:, None] * stride_qt + offs_k[None, :] * stride_qd,
                       mask=mask_t[:, None], other=0.0).to(tl.float32)
        k    = tl.load(k_base    + offs_t_chunk[:, None] * stride_kt + offs_k[None, :] * stride_kd,  # FIX 5
                       mask=mask_t[:, None], other=0.0).to(tl.float32)
        v    = tl.load(v_base    + offs_t_chunk[:, None] * stride_vt + offs_v[None, :] * stride_vd,
                       mask=mask_t[:, None], other=0.0).to(tl.float32)
        a    = tl.load(a_base    + offs_t_chunk * stride_at, mask=mask_t, other=1.0).to(tl.float32)
        b    = tl.load(b_base    + offs_t_chunk * stride_bt, mask=mask_t, other=0.0).to(tl.float32)
        dout = tl.load(dout_base + offs_t_chunk[:, None] * stride_vt + offs_v[None, :] * stride_vd,
                       mask=mask_t[:, None], other=0.0).to(tl.float32)

        if chunk_idx > 0:
            state_ptr = State + batch_idx * stride_sb + head_idx * stride_sh + (chunk_idx - 1) * stride_sc
            S_prev = tl.load(
                state_ptr + offs_k[:, None] * stride_sk + offs_v[None, :] * stride_sv,
                mask=(offs_k[:, None] < D_K) & (offs_v[None, :] < D_V), other=0.0  # FIX 6
            )
        else:
            S_prev = tl.zeros((D_K, D_V), dtype=tl.float32)  # FIX 6

        # Recompute forward intermediates
        log_a          = tl.log(a + 1e-8)
        cum_log_a      = tl.cumsum(log_a, axis=0)
        D              = tl.exp(cum_log_a[:, None] - cum_log_a[None, :])
        causal_mask    = offs_t[:, None] >= offs_t[None, :]
        D              = tl.where(causal_mask, D, 0.0)
        M              = D * b[None, :]          # Columns, not rows
        decay          = tl.exp(cum_log_a)

        mask_last      = offs_t == CHUNK_SIZE - 1  # FIX 6: was hardcoded 31
        cum_log_a_last = tl.sum(tl.where(mask_last, cum_log_a, 0.0))
        last_decay     = tl.exp(cum_log_a_last)
        D_last_row     = tl.exp(cum_log_a_last - cum_log_a)
        coeff          = D_last_row * b

        QK = tl.dot(q, tl.trans(k))

        # ── Gradient computation ──────────────────────────────────

        # Inter path
        q_S_prev      = tl.dot(q, S_prev)
        d_decay       = tl.sum(dout * q_S_prev, axis=1)
        dq_inter      = tl.dot(dout * decay[:, None], tl.trans(S_prev))
        dS_prev_inter = tl.dot(tl.trans(q), dout * decay[:, None])

        # Intra path
        d_A       = tl.dot(dout, tl.trans(v))
        d_QK      = d_A * M
        d_M       = d_A * QK

        dq_intra  = tl.dot(d_QK, k)
        dk_intra  = tl.dot(tl.trans(d_QK), q)
        dv        = tl.dot(tl.trans(QK * M), dout)

        # d_b from columns of d_M (consistent with M = D * b[None, :])
        d_D       = d_M * b[None, :]
        d_b_intra = tl.sum(d_M * D, axis=0)      # sum over rows (since b is columns)

        d_delta     = d_D * D
        d_cum_intra = tl.sum(d_delta, axis=1) - tl.sum(d_delta, axis=0)
        d_cum_intra += d_decay * decay

        # State update gradients
        dS_prev_update = dS * last_decay
        dS_prev        = dS_prev_inter + dS_prev_update
        d_last_decay   = tl.sum(dS * S_prev)

        k_weighted   = k * coeff[:, None]
        d_k_weighted = tl.dot(v, tl.trans(dS))
        d_v_update   = tl.dot(k_weighted, dS)
        dv           += d_v_update

        d_coeff    = tl.sum(d_k_weighted * k, axis=1)
        dk_update  = d_k_weighted * coeff[:, None]
        dk         = dk_intra + dk_update

        # FIX 3: d_b from coeff path also from write position
        d_b_update = d_coeff * D_last_row
        d_b        = d_b_intra + d_b_update

        d_D_last_row  = d_coeff * b
        d_cum_last    = tl.sum(d_D_last_row * D_last_row)
        d_cum_log_a   = -d_D_last_row * D_last_row

        d_cum_last    += d_last_decay * last_decay
        d_cum          = d_cum_intra + d_cum_log_a + tl.where(mask_last, d_cum_last, 0.0)

        cum_dcum   = tl.cumsum(d_cum, axis=0)
        total_dcum = tl.sum(d_cum, axis=0)
        d_log_a    = total_dcum - cum_dcum + d_cum
        d_alpha    = d_log_a / (a + 1e-8)
        dq         = dq_inter + dq_intra

        tl.store(dq_base + offs_t_chunk[:, None] * stride_qt + offs_k[None, :] * stride_qd,
                 dq, mask=mask_t[:, None])
        tl.store(dk_base + offs_t_chunk[:, None] * stride_kt + offs_k[None, :] * stride_kd,  # FIX 5
                 dk, mask=mask_t[:, None])
        tl.store(dv_base + offs_t_chunk[:, None] * stride_vt + offs_v[None, :] * stride_vd,
                 dv, mask=mask_t[:, None])
        tl.store(da_base + offs_t_chunk * stride_at, d_alpha, mask=mask_t)
        tl.store(db_base + offs_t_chunk * stride_bt, d_b,     mask=mask_t)

        dS = dS_prev

    # FIX 7: Store the final dS into dinitial_state (which is the gradient for sequence's initial state)
    if HAS_DINITIAL_STATE:
        di_ptr = Dinitial_state + batch_idx * stride_disb + head_idx * stride_dish
        tl.store(
            di_ptr + offs_k[:, None] * stride_disk + offs_v[None, :] * stride_disv,
            dS,
            mask=(offs_k[:, None] < D_K) & (offs_v[None, :] < D_V)
        )


# ──────────────────────────────────────────────────────────────────
# AUTOGRAD WRAPPER
# ──────────────────────────────────────────────────────────────────

# Default chunk size — can be overridden per-call via fused_nitro_scan(chunk_size=...)
_DEFAULT_CHUNK_SIZE = 32

class FusedNitroEngine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, state=None, chunk_size=_DEFAULT_CHUNK_SIZE):
        B, T, H, dk = q.shape
        dv = v.shape[-1]
        # FIX 6: assert power-of-2 only, not fixed size
        assert dk & (dk - 1) == 0, f"d_k must be a power of 2, got {dk}"
        assert dv & (dv - 1) == 0, f"d_v must be a power of 2, got {dv}"
        assert chunk_size & (chunk_size - 1) == 0, f"chunk_size must be a power of 2, got {chunk_size}"

        # Inputs must be float16 or bfloat16 for Triton dot to use tensor cores
        # alpha/beta stay float32 for numerical stability
        dtype = q.dtype

        q_s = q.transpose(1, 2).contiguous()
        k_s = k.transpose(1, 2).contiguous()
        v_s = v.transpose(1, 2).contiguous()
        a_s = alpha.transpose(1, 2).contiguous().float()
        b_s = beta.transpose(1, 2).contiguous().float()

        num_chunks = math.ceil(T / chunk_size)          # FIX 6
        out    = torch.empty_like(v_s)
        states = torch.empty((B, H, num_chunks, dk, dv), # FIX 6
                             device=q.device, dtype=torch.float32).contiguous()

        has_init = state is not None
        if has_init:
            is_s = state.float().contiguous()
            s_strides = is_s.stride()
        else:
            is_s      = torch.empty(0, device=q.device, dtype=torch.float32)
            s_strides = (0, 0, 0, 0)

        grid = (B, H)
        _chunk_fwd_kernel[grid](
            q_s, k_s, v_s, a_s, b_s, out, states, is_s, T,
            has_init,                          # constexpr HAS_INITIAL_STATE
            dk, dv, chunk_size,                # FIX 6: D_K, D_V, CHUNK_SIZE constexprs
            q_s.stride(0), q_s.stride(1), q_s.stride(2), q_s.stride(3),
            k_s.stride(0), k_s.stride(1), k_s.stride(2), k_s.stride(3),  # FIX 5: K strides
            v_s.stride(0), v_s.stride(1), v_s.stride(2), v_s.stride(3),
            a_s.stride(0), a_s.stride(1), a_s.stride(2),
            b_s.stride(0), b_s.stride(1), b_s.stride(2),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            s_strides[0], s_strides[1], s_strides[2], s_strides[3],
            num_warps=4,    # FIX 2: was 2, needs 4 for 64x64 matmul on Ampere
            num_stages=2,   # FIX 2: prefetch next chunk while processing current
        )

        ctx.save_for_backward(q_s, k_s, v_s, a_s, b_s, states)
        ctx.dtype      = dtype
        ctx.chunk_size = chunk_size
        ctx.has_init   = has_init                      # FIX 7
        return out.transpose(1, 2).to(dtype).contiguous(), states[:, :, -1].to(dtype)

    @staticmethod
    def backward(ctx, dout, dstate):
        q_s, k_s, v_s, a_s, b_s, states = ctx.saved_tensors
        B, H, T = q_s.shape[:3]
        dk = q_s.shape[3]
        dv = v_s.shape[3]
        chunk_size = ctx.chunk_size
        has_init = ctx.has_init

        dout_s = dout.transpose(1, 2).contiguous().float()
        dq = torch.empty_like(q_s)
        dk_ = torch.empty_like(k_s)
        dv_ = torch.empty_like(v_s)
        da = torch.empty_like(a_s)
        db = torch.empty_like(b_s)

        # FIX 7: Handle dstate parameter (gradient from the future segment)
        has_dstate = dstate is not None
        if has_dstate:
            dstate_s = dstate.float().contiguous()
            ds_strides = dstate_s.stride()
        else:
            dstate_s = torch.empty(0, device=dout.device, dtype=torch.float32)
            ds_strides = (0, 0, 0, 0)

        # FIX 7: Handle dinitial_state calculation (gradient for initial state)
        if has_init:
            dinitial_state = torch.empty((B, H, dk, dv), device=dout.device, dtype=torch.float32).contiguous()
            dis_strides = dinitial_state.stride()
        else:
            dinitial_state = torch.empty(0, device=dout.device, dtype=torch.float32)
            dis_strides = (0, 0, 0, 0)

        grid = (B, H)
        _chunk_bwd_kernel[grid](
            q_s, k_s, v_s, a_s, b_s, states, dout_s,
            dq, dk_, dv_, da, db,
            dstate_s, dinitial_state,          # FIX 7
            T,
            dk, dv, chunk_size,                # FIX 6: D_K, D_V, CHUNK_SIZE constexprs
            has_dstate, has_init,              # FIX 7
            q_s.stride(0), q_s.stride(1), q_s.stride(2), q_s.stride(3),
            k_s.stride(0), k_s.stride(1), k_s.stride(2), k_s.stride(3),  # FIX 5: K strides
            v_s.stride(0), v_s.stride(1), v_s.stride(2), v_s.stride(3),
            a_s.stride(0), a_s.stride(1), a_s.stride(2),
            b_s.stride(0), b_s.stride(1), b_s.stride(2),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            ds_strides[0], ds_strides[1], ds_strides[2], ds_strides[3],       # FIX 7
            dis_strides[0], dis_strides[1], dis_strides[2], dis_strides[3],   # FIX 7
            num_warps=4,
            num_stages=2,
        )

        dtype = ctx.dtype
        return (
            dq.transpose(1,2).to(dtype),
            dk_.transpose(1,2).to(dtype),
            dv_.transpose(1,2).to(dtype),
            da.transpose(1,2).to(dtype),
            db.transpose(1,2).to(dtype),
            dinitial_state.to(dtype) if has_init else None,  # FIX 7: return dinitial_state
            None,   # chunk_size has no gradient
        )


def fused_nitro_scan(q, k, v, alpha, beta, state=None, chunk_size=_DEFAULT_CHUNK_SIZE):
    """
    Drop-in replacement for chunkwise_hgdm_forward.
    Works on any CUDA GPU with Triton support (no external libraries).
    RTX 3090 Ti: Ampere cc8.6, fully compatible.

    Args:
        q, k, v  : (B, T, H, d_k/d_v) — must be power-of-2 last dim
        alpha, beta: (B, T, H)
        state    : optional initial state (B, H, d_k, d_v)
        chunk_size: int, must be power of 2 (default 32)
    """
    return FusedNitroEngine.apply(q, k, v, alpha, beta, state, chunk_size)