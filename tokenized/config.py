import math
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class TokenizedConfig:
    # Model architecture
    vocab_size: int = 32000              # Default 32k BPE tokenizer
    d_model: int = 384                  # Hidden size
    n_layers: int = 6                   # Number of recurrent/attention layers
    n_heads: int = 6                    # Number of heads (d_k = d_model // n_heads = 64)
    d_k: int = 64                       # Key dimension per head
    d_v: int = 64                       # Value dimension per head
    d_ff: int = 1536                    # SwiGLU intermediate dimension
    max_seq_len: int = 2048             # Context window
    dropout: float = 0.0                # Pretraining dropout
    
    # HGDM specific parameters
    use_variable_delta_t: bool = True   # Time-based continuous ODE decay (CfC)
    use_epistemic_gate: bool = True     # State confidence gating via tanh(||n||)
    sparse_write_threshold: float = 0.1 # Sparsity threshold for ReLU beta gate
    chunk_size: int = 32                # Nitro parallel scan tile size (GPU SRAM)
    
    # Chinchilla scaling configuration
    # Optimal tokens ~= 20 * parameters
    chinchilla_ratio: float = 20.0
    
    def count_parameters(self) -> int:
        """Calculates total parameter count assuming tied embeddings."""
        # Embedding (tied with lm_head): vocab_size * d_model
        embed_params = self.vocab_size * self.d_model
        
        # Per layer:
        # RMSNorm (2 per layer): 2 * d_model
        norm_params = 2 * self.d_model
        
        # SwiGLU: w1 (d_model -> d_ff), w2 (d_model -> d_ff), w3 (d_ff -> d_model)
        swiglu_params = 3 * self.d_model * self.d_ff
        
        # Mixer (HGDM or MHA):
        # Q, K, V projections + Output projection = 4 * d_model^2
        # Gating projections: W_delta, W_beta, etc. ~= 2 * d_model
        mixer_params = 4 * (self.d_model ** 2) + 2 * self.d_model
        
        layer_params = (norm_params + swiglu_params + mixer_params) * self.n_layers
        final_norm = self.d_model
        
        return embed_params + layer_params + final_norm

    def target_training_tokens(self) -> int:
        """Returns the Chinchilla optimal training token count."""
        return int(self.count_parameters() * self.chinchilla_ratio)

    @classmethod
    def get_tier(cls, tier: str = "pilot", vocab_size: int = 32000) -> "TokenizedConfig":
        """Pre-configured scale tiers."""
        if tier == "pilot":
            # ~15M total params (with 32k vocab, ~12M in tied embed, ~3.5M in layers)
            # Target tokens: ~300M tokens
            return cls(
                vocab_size=vocab_size,
                d_model=384,
                n_layers=6,
                n_heads=6,
                d_k=64,
                d_v=64,
                d_ff=1024,
                max_seq_len=2048,
            )
        elif tier == "standard":
            # ~25M params (d=512, L=8, H=8)
            # Target tokens: ~500M tokens
            return cls(
                vocab_size=vocab_size,
                d_model=512,
                n_layers=8,
                n_heads=8,
                d_k=64,
                d_v=64,
                d_ff=1536,
                max_seq_len=2048,
            )
        elif tier == "paper":
            # ~120M params matching paper baseline
            # Target tokens: ~2.4B tokens
            return cls(
                vocab_size=vocab_size,
                d_model=768,
                n_layers=12,
                n_heads=12,
                d_k=64,
                d_v=64,
                d_ff=3072,
                max_seq_len=2048,
            )
        else:
            raise ValueError(f"Unknown tier: {tier}. Choose 'pilot', 'standard', or 'paper'.")
