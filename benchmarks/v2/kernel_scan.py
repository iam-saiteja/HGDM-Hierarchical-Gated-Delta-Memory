import torch
import torch.nn.functional as F

def ssd_parallel_scan(q, k, v, alpha, beta, chunk_size=64):
    """
    State Space Duality (SSD) Chunkwise Formulation:
    Breaks the O(T^2) mask into tiny O(C^2) chunks.
    Processes the chunks in parallel on Tensor Cores, and passes the memory matrix
    sequentially between chunks. Achieves blazing fast O(N) training speed
    without the need for custom Triton C++ compilation.
    """
    B, T, H, d_k = q.shape
    d_v = v.shape[-1]
    
    alpha = alpha.squeeze(-1).squeeze(-1).transpose(1, 2) # [B, H, T]
    beta = beta.squeeze(-1).squeeze(-1).transpose(1, 2)   # [B, H, T]
    
    q = q.transpose(1, 2) # [B, H, T, d_k]
    k = k.transpose(1, 2) # [B, H, T, d_k]
    v = v.transpose(1, 2) # [B, H, T, d_v]
    
    # Pad to chunk size if necessary
    pad_len = (chunk_size - (T % chunk_size)) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        alpha = F.pad(alpha, (0, pad_len), value=1.0)
        beta = F.pad(beta, (0, pad_len), value=0.0)
        T = T + pad_len
        
    num_chunks = T // chunk_size
    
    q_c = q.view(B, H, num_chunks, chunk_size, d_k)
    k_c = k.view(B, H, num_chunks, chunk_size, d_k)
    v_c = v.view(B, H, num_chunks, chunk_size, d_v)
    alpha_c = alpha.view(B, H, num_chunks, chunk_size)
    beta_c = beta.view(B, H, num_chunks, chunk_size)
    
    # 1. Compute intra-chunk decays
    log_alpha_c = torch.log(alpha_c.clamp(min=1e-8))
    C_c = torch.cumsum(log_alpha_c, dim=-1) # [B, H, num_chunks, chunk_size]
    
    # 2. Build tiny Chunkwise Mask
    log_M_c = C_c.unsqueeze(-1) - C_c.unsqueeze(-2) # [B, H, num_chunks, C, C]
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool))
    log_M_c = log_M_c.masked_fill(~causal_mask, float('-inf'))
    M_c = torch.exp(log_M_c) # [B, H, num_chunks, C, C]
    M_beta_c = M_c * beta_c.unsqueeze(-2)
    
    # 3. Intra-chunk attention (Compute interactions strictly inside the chunk)
    attn_c = torch.matmul(q_c, k_c.transpose(-1, -2)) * M_beta_c
    out_intra = torch.matmul(attn_c, v_c) # [B, H, num_chunks, C, d_v]
    
    # 4. Pre-compute the chunk's state update (for inter-chunk passing)
    decay_total = torch.exp(C_c[:, :, :, -1]) # [B, H, num_chunks]
    last_decay_c = M_beta_c[:, :, :, -1, :].unsqueeze(-1) # [B, H, num_chunks, C, 1]
    k_v_outer_c = k_c.unsqueeze(-1) * v_c.unsqueeze(-2) # [B, H, num_chunks, C, d_k, d_v]
    chunk_state_update = (last_decay_c.unsqueeze(-1) * k_v_outer_c).sum(dim=3) # [B, H, num_chunks, d_k, d_v]
    
    # 5. Inter-chunk sequential scan (Lightning fast python loop over chunks)
    S = torch.zeros(B, H, d_k, d_v, device=q.device, dtype=q.dtype)
    out_inter = []
    
    for i in range(num_chunks):
        # Decay the state to each position within the chunk
        decay_to_pos = torch.exp(C_c[:, :, i, :]).unsqueeze(-1).unsqueeze(-1) # [B, H, C, 1, 1]
        S_read = S.unsqueeze(2) * decay_to_pos # [B, H, C, d_k, d_v]
        
        # Read out the memory using queries
        out_s = torch.einsum('bhcd,bhcde->bhce', q_c[:, :, i], S_read)
        out_inter.append(out_s)
        
        # Advance the state matrix S for the next chunk
        S = S * decay_total[:, :, i].unsqueeze(-1).unsqueeze(-1) + chunk_state_update[:, :, i]
        
    out_inter = torch.stack(out_inter, dim=2) # [B, H, num_chunks, C, d_v]
    
    # 6. Recombine and format
    out = out_intra + out_inter
    out = out.view(B, H, T, d_v)
    
    # Remove padding if it was added
    if pad_len > 0:
        out = out[:, :, :-pad_len, :]
        
    return out.transpose(1, 2).contiguous(), S
