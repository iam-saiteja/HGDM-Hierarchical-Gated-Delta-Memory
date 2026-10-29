import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from tokenized.config import TokenizedConfig
from tokenized.model_hgdm import RMSNorm, SwiGLU


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TokenizedConfig):
        super().__init__()
        self.H = config.n_heads
        self.d_k = config.d_k
        self.d_model = config.d_model
        
        # Q, K, V projections
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.dropout = config.dropout

    def forward(
        self, 
        x: torch.Tensor, 
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        
        # [B, H, T, d_k]
        q = self.W_q(x).view(B, T, self.H, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.H, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.H, self.d_k).transpose(1, 2)
        
        # KV Cache for autoregressive generation
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
            
        new_kv_cache = (k, v)
        
        # Scaled dot-product attention (uses FlashAttention-2 / SDPA under the hood)
        # Is causal if generating without cache, or single step when cached
        is_causal = (T > 1 and kv_cache is None)
        out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out), new_kv_cache


class MatchedTransformerLayer(nn.Module):
    def __init__(self, config: TokenizedConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config.d_model, config.d_ff, dropout=config.dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, next_kv = self.attn(self.norm1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, next_kv


class MatchedTransformer(nn.Module):
    """
    Matched Baseline Transformer.
    - Exactly matched parameter count to TokenizedHGDM.
    - Uses standard causal self-attention with KV-Cache.
    - Pre-RMSNorm and SwiGLU.
    - Tied input embedding and output projection.
    """
    def __init__(self, config: TokenizedConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        self.layers = nn.ModuleList([
            MatchedTransformerLayer(config) for _ in range(config.n_layers)
        ])
        
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Weight tying
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        pos_offset: int = 0,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T = input_ids.shape
        device = input_ids.device
        
        pos = torch.arange(pos_offset, pos_offset + T, device=device).unsqueeze(0)
        pos = pos.clamp(max=self.config.max_seq_len - 1)
        
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        
        next_kv_caches = []
        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, next_kv = layer(x, kv_cache=cache_i)
            next_kv_caches.append(next_kv)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, next_kv_caches

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Autoregressive Generation with Standard KV-Cache.
        (Shows the growing O(T) memory footprint of Transformers).
        """
        self.eval()
        B, T_prompt = prompt_ids.shape
        device = prompt_ids.device
        
        # 1. Forward prompt to populate initial KV caches
        logits, kv_caches = self(prompt_ids, pos_offset=0)
        
        generated = [prompt_ids]
        curr_token = prompt_ids[:, -1:]
        
        for step in range(max_new_tokens):
            pos_offset = T_prompt + step
            logits, kv_caches = self(curr_token, kv_caches=kv_caches, pos_offset=pos_offset)
            logits_step = logits[:, -1, :] / max(1e-5, temperature)
            
            if top_k > 0:
                v, _ = torch.topk(logits_step, min(top_k, logits_step.size(-1)))
                logits_step[logits_step < v[:, [-1]]] = -float('inf')
                
            probs = F.softmax(logits_step, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated.append(next_token)
            curr_token = next_token
            
        return torch.cat(generated, dim=1)
