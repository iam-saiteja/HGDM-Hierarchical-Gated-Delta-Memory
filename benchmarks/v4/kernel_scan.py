import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def chunkwise_hgdm_forward(q, k, v, alpha, beta, chunk_size=128):
    """
    V6 Production Engine: Stable Chunkwise Parallelism.
    Uses explicit dimension-based padding for 4D and 5D tensors.
    """
    B, T, H, d_k = q.shape
    d_v = v.shape[-1]
    
    # 1. Explicit Padding to Chunk Size
    pad_len = (chunk_size - (T % chunk_size)) % chunk_size
    if pad_len > 0:
        # Pad the Time dimension (Dim 1) for all tensors
        # F.pad format is (last_dim_front, last_dim_back, prev_dim_front, prev_dim_back, ...)
        # For [B, T, H, D], T is index 1. We need to pad (0,0, 0,0, 0,pad_len, 0,0)
        q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        
        # For [B, T, H, 1, 1], T is index 1. We need to pad (0,0, 0,0, 0,0, 0,pad_len, 0,0)
        alpha = F.pad(alpha, (0, 0, 0, 0, 0, 0, 0, pad_len))
        beta = F.pad(beta, (0, 0, 0, 0, 0, 0, 0, pad_len))
    
    T_pad = q.shape[1]
    num_chunks = T_pad // chunk_size
    
    # 2. Stable Reshaping [B, T, H, D] -> [B, H, num_chunks, C, D]
    q_c = q.permute(0, 2, 1, 3).contiguous().view(B, H, num_chunks, chunk_size, d_k)
    k_c = k.permute(0, 2, 1, 3).contiguous().view(B, H, num_chunks, chunk_size, d_k)
    v_c = v.permute(0, 2, 1, 3).contiguous().view(B, H, num_chunks, chunk_size, d_v)
    
    # Gates: [B, T, H, 1, 1] -> [B, H, num_chunks, C]
    alpha_c = alpha.squeeze(-1).squeeze(-1).permute(0, 2, 1).contiguous().view(B, H, num_chunks, chunk_size)
    beta_c = beta.squeeze(-1).squeeze(-1).permute(0, 2, 1).contiguous().view(B, H, num_chunks, chunk_size)
    
    # --- The Custom Forward Function for Checkpointing ---
    def process_chunk(q_i, k_i, v_i, alpha_i, beta_i, S_prev):
        """Processes one chunk with vectorized matrix math."""
        log_a = torch.log(alpha_i.clamp(min=1e-8))
        cum_log_a = torch.cumsum(log_a, dim=-1)
        log_M = cum_log_a.unsqueeze(-1) - cum_log_a.unsqueeze(-2)
        mask = torch.tril(torch.ones(chunk_size, chunk_size, device=q_i.device, dtype=torch.bool))
        log_M = log_M.masked_fill(~mask, float('-inf'))
        M = torch.exp(log_M) * beta_i.unsqueeze(-2)
        out_intra = torch.matmul(torch.matmul(q_i, k_i.transpose(-1, -2)) * M, v_i)
        decay_factors = torch.exp(cum_log_a).unsqueeze(-1)
        out_inter = torch.matmul(q_i, S_prev) * decay_factors
        out_chunk = out_intra + out_inter
        total_chunk_decay = torch.exp(cum_log_a[:, :, -1]).unsqueeze(-1).unsqueeze(-1)
        last_decay_M = M[:, :, -1, :].unsqueeze(-1)
        k_beta = k_i * last_decay_M
        chunk_update = torch.matmul(k_beta.transpose(-1, -2), v_i)
        S_new = (S_prev * total_chunk_decay) + chunk_update
        return out_chunk, S_new

    # Loop over chunks using Gradient Checkpointing
    S = torch.zeros(B, H, d_k, d_v, device=q.device, dtype=q.dtype)
    outputs = []
    
    for i in range(num_chunks):
        out_c, S = checkpoint(
            process_chunk, 
            q_c[:, :, i], k_c[:, :, i], v_c[:, :, i], alpha_c[:, :, i], beta_c[:, :, i], S,
            use_reentrant=False
        )
        outputs.append(out_c)
        
    out = torch.stack(outputs, dim=2)
    out = out.permute(0, 2, 3, 1, 4).contiguous().view(B, T_pad, H, d_v)
    if pad_len > 0:
        out = out[:, :-pad_len, :, :]
    return out, S
