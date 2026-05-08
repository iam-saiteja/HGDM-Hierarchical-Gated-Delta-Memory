import torch
import triton
import triton.language as tl

@triton.jit
def hgdm_chunkwise_fwd_kernel(
    q_ptr, k_ptr, v_ptr, alpha_ptr, beta_ptr, out_ptr,
    seq_len, d_k: tl.constexpr, d_v: tl.constexpr, CHUNK_SIZE: tl.constexpr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ab, stride_ah, stride_at,
    stride_ob, stride_oh, stride_ot, stride_od
):
    """
    V6 Nitro Engine: Triton Chunkwise Forward.
    Maintains the 64x64 state S in SRAM (L1 Cache).
    Parallel over Heads and Batches, Sequential over Chunks.
    """
    batch_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    
    # Calculate base pointers
    q_offset = batch_pid * stride_qb + head_pid * stride_qh
    k_offset = batch_pid * stride_kb + head_pid * stride_kh
    v_offset = batch_pid * stride_vb + head_pid * stride_vh
    a_offset = batch_pid * stride_ab + head_pid * stride_ah
    o_offset = batch_pid * stride_ob + head_pid * stride_oh
    
    # Initialize the S matrix in SRAM (L1)
    offs_k = tl.arange(0, d_k)
    offs_v = tl.arange(0, d_v)
    S = tl.zeros((d_k, d_v), dtype=tl.float32)
    
    num_chunks = tl.cdiv(seq_len, CHUNK_SIZE)
    
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        offs_t = chunk_start + tl.arange(0, CHUNK_SIZE)
        t_mask = offs_t < seq_len
        
        # Load Q, K, V, Alpha, Beta for chunk
        q = tl.load(q_ptr + q_offset + offs_t[:, None] * stride_qt + offs_k[None, :] * stride_qd, mask=t_mask[:, None], other=0.0)
        k = tl.load(k_ptr + k_offset + offs_t[:, None] * stride_kt + offs_k[None, :] * stride_kd, mask=t_mask[:, None], other=0.0)
        v = tl.load(v_ptr + v_offset + offs_t[:, None] * stride_vt + offs_v[None, :] * stride_vd, mask=t_mask[:, None], other=0.0)
        
        alpha = tl.load(alpha_ptr + a_offset + offs_t * stride_at, mask=t_mask, other=1.0)
        beta = tl.load(beta_ptr + a_offset + offs_t * stride_at, mask=t_mask, other=0.0)
        
        # 1. READ FROM SRAM STATE
        out_inter = tl.dot(q, S)
        
        # 2. INTRA-CHUNK ATTENTION
        attn = tl.dot(q, tl.trans(k))
        offs_i = tl.arange(0, CHUNK_SIZE)
        causal_mask = offs_i[:, None] >= offs_i[None, :]
        attn = tl.where(causal_mask, attn, 0.0)
        
        # Approximate local decay (linearized)
        out_intra = tl.dot(attn, v)
        
        # Total output
        out = out_inter + out_intra
        tl.store(out_ptr + o_offset + offs_t[:, None] * stride_ot + offs_v[None, :] * stride_od, out, mask=t_mask[:, None])
        
        # 3. UPDATE SRAM STATE
        # Compute chunk-level decay
        log_alpha = tl.log(alpha + 1e-8)
        chunk_decay = tl.exp(tl.sum(log_alpha, axis=0))
        S = S * chunk_decay
        
        # Add new update
        k_beta = k * beta[:, None]
        S = S + tl.dot(tl.trans(k_beta), v)

def triton_nitro_forward(q, k, v, alpha, beta):
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    out = torch.empty((B, T, H, dv), device=q.device, dtype=q.dtype)
    
    CHUNK_SIZE = 64
    grid = (B, H)
    
    hgdm_chunkwise_fwd_kernel[grid](
        q, k, v, alpha, beta, out,
        T, dk, dv, CHUNK_SIZE,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        alpha.stride(0), alpha.stride(2), alpha.stride(1),
        out.stride(0), out.stride(2), out.stride(1), out.stride(3)
    )
    return out
