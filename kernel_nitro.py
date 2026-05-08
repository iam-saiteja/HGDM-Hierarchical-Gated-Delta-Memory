import torch
import triton
import triton.language as tl
import math

# -----------------------------------------------------------
# Forward kernel (chunk size = 32)
# -----------------------------------------------------------
@triton.jit
def _chunk_fwd_kernel(
    Q, K, V, Alpha, Beta, Out, State,
    Initial_State, # New argument
    seq_len,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ab, stride_ah, stride_at,
    stride_bb, stride_bh, stride_bt,
    stride_sb, stride_sh, stride_sc, stride_sk, stride_sv,
    stride_isb, stride_ish, stride_isk, stride_isv, # Strides for initial state
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    q_base = Q + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K + batch_idx * stride_qb + head_idx * stride_qh
    v_base = V + batch_idx * stride_vb + head_idx * stride_vh
    a_base = Alpha + batch_idx * stride_ab + head_idx * stride_ah
    b_base = Beta  + batch_idx * stride_bb + head_idx * stride_bh
    o_base = Out   + batch_idx * stride_vb + head_idx * stride_vh

    offs_k = tl.arange(0, 64)        # d_k = 64
    offs_v = tl.arange(0, 64)        # d_v = 64
    offs_t = tl.arange(0, 32)        # chunk size = 32

    # Load initial state if provided
    if Initial_State is not None:
        is_ptr = Initial_State + batch_idx * stride_isb + head_idx * stride_ish
        S = tl.load(is_ptr + offs_k[:, None] * stride_isk + offs_v[None, :] * stride_isv)
    else:
        S = tl.zeros((64, 64), dtype=tl.float32)
        
    num_chunks = tl.cdiv(seq_len, 32)
    # ... (rest of kernel)

    for chunk_idx in range(num_chunks):
        t_start = chunk_idx * 32
        offs_t_chunk = t_start + offs_t
        mask_t = offs_t_chunk < seq_len

        q = tl.load(q_base + offs_t_chunk[:, None] * stride_qt + offs_k[None, :] * stride_qd,
                    mask=mask_t[:, None], other=0.0)
        k = tl.load(k_base + offs_t_chunk[:, None] * stride_qt + offs_k[None, :] * stride_qd,
                    mask=mask_t[:, None], other=0.0)
        v = tl.load(v_base + offs_t_chunk[:, None] * stride_vt + offs_v[None, :] * stride_vd,
                    mask=mask_t[:, None], other=0.0)
        a = tl.load(a_base + offs_t_chunk * stride_at, mask=mask_t, other=1.0)
        b = tl.load(b_base + offs_t_chunk * stride_bt, mask=mask_t, other=0.0)

        log_a = tl.log(a.to(tl.float32) + 1e-8)
        cum_log_a = tl.cumsum(log_a, axis=0)                 # (32,)

        D = tl.exp(cum_log_a[:, None] - cum_log_a[None, :])  # (32,32)
        causal_mask = offs_t[:, None] >= offs_t[None, :]
        D = tl.where(causal_mask, D, 0.0)
        M = D * b[None, :].to(tl.float32)

        QK = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32)))   # (32,32)
        out_intra = tl.dot(QK * M, v.to(tl.float32))                # (32,64)

        decay = tl.exp(cum_log_a)                                    # (32,)
        out_inter = tl.dot(q.to(tl.float32), S) * decay[:, None]    # (32,64)

        out = (out_intra + out_inter).to(q.dtype)
        tl.store(o_base + offs_t_chunk[:, None] * stride_vt + offs_v[None, :] * stride_vd,
                 out, mask=mask_t[:, None])

        # State update
        last_idx = 31
        mask_last = offs_t == last_idx
        cum_log_a_last = tl.sum(tl.where(mask_last, cum_log_a, 0.0))
        last_decay = tl.exp(cum_log_a_last)

        D_last_row = tl.exp(cum_log_a_last - cum_log_a)      # (32,)
        coeff = D_last_row * b                                # (32,)
        k_weighted = (k.to(tl.float32) * coeff[:, None])      # (32,64)
        chunk_update = tl.dot(tl.trans(k_weighted), v.to(tl.float32))  # (64,64)

        S = S * last_decay + chunk_update

        state_ptr = State + batch_idx * stride_sb + head_idx * stride_sh + chunk_idx * stride_sc
        tl.store(state_ptr + offs_k[:, None] * stride_sk + offs_v[None, :] * stride_sv,
                 S, mask=(offs_k[:, None] < 64) & (offs_v[None, :] < 64))


# -----------------------------------------------------------
# Backward kernel (chunk size = 32, all @ replaced by tl.dot)
# -----------------------------------------------------------
@triton.jit
def _chunk_bwd_kernel(
    Q, K, V, Alpha, Beta, State, Dout,
    DQ, DK, DV, DAlpha, DBeta,
    seq_len,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ab, stride_ah, stride_at,
    stride_bb, stride_bh, stride_bt,
    stride_sb, stride_sh, stride_sc, stride_sk, stride_sv,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    q_base   = Q    + batch_idx * stride_qb + head_idx * stride_qh
    k_base   = K    + batch_idx * stride_qb + head_idx * stride_qh
    v_base   = V    + batch_idx * stride_vb + head_idx * stride_vh
    a_base   = Alpha + batch_idx * stride_ab + head_idx * stride_ah
    b_base   = Beta  + batch_idx * stride_bb + head_idx * stride_bh
    dout_base = Dout + batch_idx * stride_vb + head_idx * stride_vh

    dq_base = DQ + batch_idx * stride_qb + head_idx * stride_qh
    dk_base = DK + batch_idx * stride_qb + head_idx * stride_qh
    dv_base = DV + batch_idx * stride_vb + head_idx * stride_vh
    da_base = DAlpha + batch_idx * stride_ab + head_idx * stride_ah
    db_base = DBeta  + batch_idx * stride_bb + head_idx * stride_bh

    offs_k = tl.arange(0, 64)
    offs_v = tl.arange(0, 64)
    offs_t = tl.arange(0, 32)

    dS = tl.zeros((64, 64), dtype=tl.float32)
    num_chunks = tl.cdiv(seq_len, 32)

    for chunk_idx in range(num_chunks - 1, -1, -1):
        t_start = chunk_idx * 32
        offs_t_chunk = t_start + offs_t
        mask_t = offs_t_chunk < seq_len

        q = tl.load(q_base + offs_t_chunk[:, None] * stride_qt + offs_k[None, :] * stride_qd,
                    mask=mask_t[:, None], other=0.0).to(tl.float32)
        k = tl.load(k_base + offs_t_chunk[:, None] * stride_qt + offs_k[None, :] * stride_qd,
                    mask=mask_t[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_base + offs_t_chunk[:, None] * stride_vt + offs_v[None, :] * stride_vd,
                    mask=mask_t[:, None], other=0.0).to(tl.float32)
        a = tl.load(a_base + offs_t_chunk * stride_at, mask=mask_t, other=1.0).to(tl.float32)
        b = tl.load(b_base + offs_t_chunk * stride_bt, mask=mask_t, other=0.0).to(tl.float32)
        dout = tl.load(dout_base + offs_t_chunk[:, None] * stride_vt + offs_v[None, :] * stride_vd,
                       mask=mask_t[:, None], other=0.0).to(tl.float32)

        if chunk_idx > 0:
            state_ptr = State + batch_idx * stride_sb + head_idx * stride_sh + (chunk_idx - 1) * stride_sc
            S_prev = tl.load(state_ptr + offs_k[:, None]*stride_sk + offs_v[None, :]*stride_sv,
                             mask=(offs_k[:, None] < 64) & (offs_v[None, :] < 64), other=0.0)
        else:
            S_prev = tl.zeros((64, 64), dtype=tl.float32)

        # Recompute forward intermediates (all shapes now 32 internal)
        log_a = tl.log(a + 1e-8)
        cum_log_a = tl.cumsum(log_a, axis=0)
        D = tl.exp(cum_log_a[:, None] - cum_log_a[None, :])   # (32,32)
        causal_mask = offs_t[:, None] >= offs_t[None, :]
        D = tl.where(causal_mask, D, 0.0)
        M = D * b[None, :]                                      # (32,32)
        decay = tl.exp(cum_log_a)                               # (32,)

        last_idx = 31
        mask_last = offs_t == last_idx
        cum_log_a_last = tl.sum(tl.where(mask_last, cum_log_a, 0.0))
        last_decay = tl.exp(cum_log_a_last)
        D_last_row = tl.exp(cum_log_a_last - cum_log_a)        # (32,)
        coeff = D_last_row * b

        QK = tl.dot(q, tl.trans(k))                             # (32,32)

        # ========== Gradients ==========
        d_intra = dout
        d_inter = dout

        # Inter path
        q_S_prev = tl.dot(q, S_prev)                                     # (32,64)
        d_decay = tl.sum(d_inter * q_S_prev, axis=1)                     # (32,)
        dq_inter = tl.dot(d_inter * decay[:, None], tl.trans(S_prev))    # (32,64)
        dS_prev_inter = tl.dot(tl.trans(q), d_inter * decay[:, None])    # (64,64)

        # Intra path
        d_A = tl.dot(d_intra, tl.trans(v))                               # (32,32)
        d_QK = d_A * M
        d_M = d_A * QK

        dq_intra = tl.dot(d_QK, k)                                       # (32,64)
        dk_intra = tl.dot(tl.trans(d_QK), q)                             # (32,64)
        dv = tl.dot(tl.trans(QK * M), d_intra)                           # (32,64)

        d_D = d_M * b[None, :]
        d_b_intra = tl.sum(d_M * D, axis=0)

        d_delta = d_D * D
        d_cum_intra = tl.sum(d_delta, axis=1) - tl.sum(d_delta, axis=0)
        d_cum_intra += d_decay * decay

        # State update gradients
        dS_prev_update = dS * last_decay
        dS_prev = dS_prev_inter + dS_prev_update

        d_last_decay = tl.sum(dS * S_prev)

        k_weighted = k * coeff[:, None]                                  # (32,64)
        d_k_weighted = tl.dot(v, tl.trans(dS))                           # (32,64)
        d_v_update = tl.dot(k_weighted, dS)                              # (32,64)
        dv += d_v_update

        d_coeff = tl.sum(d_k_weighted * k, axis=1)                       # (32,)
        dk_update = d_k_weighted * coeff[:, None]
        dk = dk_intra + dk_update

        d_b_update = d_coeff * D_last_row
        d_b = d_b_intra + d_b_update

        d_D_last_row = d_coeff * b
        d_cum_last = tl.sum(d_D_last_row * D_last_row)
        d_cum_log_a = -d_D_last_row * D_last_row

        d_cum_last += d_last_decay * last_decay
        d_cum = d_cum_intra + d_cum_log_a + tl.where(mask_last, d_cum_last, 0.0)

        cum_dcum = tl.cumsum(d_cum, axis=0)
        total_dcum = tl.sum(d_cum, axis=0)
        d_log_a = total_dcum - cum_dcum + d_cum

        d_alpha = d_log_a / (a + 1e-8)
        dq = dq_inter + dq_intra

        # Store
        tl.store(dq_base + offs_t_chunk[:, None] * stride_qt + offs_k[None, :] * stride_qd, dq, mask=mask_t[:, None])
        tl.store(dk_base + offs_t_chunk[:, None] * stride_qt + offs_k[None, :] * stride_qd, dk, mask=mask_t[:, None])
        tl.store(dv_base + offs_t_chunk[:, None] * stride_vt + offs_v[None, :] * stride_vd, dv, mask=mask_t[:, None])
        tl.store(da_base + offs_t_chunk * stride_at, d_alpha, mask=mask_t)
        tl.store(db_base + offs_t_chunk * stride_bt, d_b, mask=mask_t)

        dS = dS_prev


# -----------------------------------------------------------
# Autograd wrapper (chunk_size=32)
# -----------------------------------------------------------
class FusedNitroEngine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, state=None, chunk_size=32):
        B, T, H, dk = q.shape
        dv = v.shape[-1]
        assert dk == 64 and dv == 64, f"HGDM requires d_k=d_v=64, got {dk},{dv}"

        q_s = q.transpose(1, 2).contiguous()
        k_s = k.transpose(1, 2).contiguous()
        v_s = v.transpose(1, 2).contiguous()
        a_s = alpha.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous()
        b_s = beta.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous()

        num_chunks = math.ceil(T / 32)
        out = torch.empty_like(v_s)
        states = torch.empty((B, H, num_chunks, 64, 64), device=q.device, dtype=torch.float32).contiguous()

        # Handle Initial State
        if state is not None:
            is_s = state.contiguous()
            stride_isb, stride_ish, stride_isk, stride_isv = is_s.stride()
        else:
            is_s = None
            stride_isb, stride_ish, stride_isk, stride_isv = 0, 0, 0, 0

        grid = (B, H)
        _chunk_fwd_kernel[grid](
            q_s, k_s, v_s, a_s, b_s, out, states,
            is_s, T,
            q_s.stride(0), q_s.stride(1), q_s.stride(2), q_s.stride(3),
            v_s.stride(0), v_s.stride(1), v_s.stride(2), v_s.stride(3),
            a_s.stride(0), a_s.stride(1), a_s.stride(2),
            b_s.stride(0), b_s.stride(1), b_s.stride(2),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            stride_isb, stride_ish, stride_isk, stride_isv,
            num_warps=2, num_stages=1
        )

        ctx.save_for_backward(q_s, k_s, v_s, a_s, b_s, states)
        out_t = out.transpose(1, 2).contiguous()
        return out_t, states[:, :, -1, :, :]

    @staticmethod
    def backward(ctx, dout, dstate):
        q_s, k_s, v_s, a_s, b_s, states = ctx.saved_tensors
        B, H, T = q_s.shape[0], q_s.shape[1], q_s.shape[2]

        dout_s = dout.transpose(1, 2).contiguous().to(q_s.dtype)
        dq = torch.empty_like(q_s)
        dk = torch.empty_like(k_s)
        dv = torch.empty_like(v_s)
        da = torch.empty_like(a_s)
        db = torch.empty_like(b_s)

        grid = (B, H)
        _chunk_bwd_kernel[grid](
            q_s, k_s, v_s, a_s, b_s, states, dout_s,
            dq, dk, dv, da, db,
            T,
            q_s.stride(0), q_s.stride(1), q_s.stride(2), q_s.stride(3),
            v_s.stride(0), v_s.stride(1), v_s.stride(2), v_s.stride(3),
            a_s.stride(0), a_s.stride(1), a_s.stride(2),
            b_s.stride(0), b_s.stride(1), b_s.stride(2),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            num_warps=2, num_stages=1
        )

        return (dq.transpose(1,2), dk.transpose(1,2), dv.transpose(1,2),
                da.unsqueeze(-1).unsqueeze(-1).transpose(1,2),
                db.unsqueeze(-1).unsqueeze(-1).transpose(1,2),
                None)

def fused_nitro_scan(q, k, v, alpha, beta, state=None):
    return FusedNitroEngine.apply(q, k, v, alpha, beta, state, 32)