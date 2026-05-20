import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
from hgdm_ultimate import HGDMLayer, RMSNorm, HGDMConfig

@dataclass
class OmegaConfig:
    # Byte-Catcher / Renderer (High Frequency, Low Mass)
    d_byte: int = 256
    catcher_layers: int = 2
    renderer_layers: int = 2
    
    # Semantic Core (Low Frequency, High Mass - where the 1B params live)
    d_model: int = 2048
    core_layers: int = 20
    n_heads: int = 16
    d_k: int = 64
    d_v: int = 64
    d_ff: int = 8192
    
    # Decimation Physics
    decimation_rate: int = 8  # W = 8
    vocab_size: int = 256
    max_position_embeddings: int = 65536  # Massive context
    
    use_variable_delta_t: bool = False
    use_state_fusion: bool = False

class CausalDecimator(nn.Module):
    """Compresses T bytes into T/W semantic chunks safely."""
    def __init__(self, config: OmegaConfig):
        super().__init__()
        self.W = config.decimation_rate
        # Trap C Fix: Softmax Causal Decay
        # Initialize bias to favor the most recent byte (index W-1)
        self.lambda_decay = nn.Parameter(torch.tensor(0.5)) 
        self.proj = nn.Linear(config.d_byte, config.d_model, bias=False)
        self.norm = RMSNorm(config.d_byte)

    def forward(self, x):
        B, T, D = x.shape
        # Pad sequence if not perfectly divisible by W
        pad_len = (self.W - (T % self.W)) % self.W
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        N = x.shape[1] // self.W
        x_grouped = x.view(B, N, self.W, D) # [B, N, W, d_byte]
        
        # Softmax normalized causal decay weights
        indices = torch.arange(self.W, device=x.device, dtype=torch.float32)
        # Reverse indices so the highest weight lands on the most recent byte in the window
        causal_distances = (self.W - 1) - indices 
        weights = torch.softmax(-F.softplus(self.lambda_decay) * causal_distances, dim=-1)
        
        # Integrate and project
        x_integrated = (x_grouped * weights.view(1, 1, self.W, 1)).sum(dim=2) # [B, N, d_byte]
        return self.proj(self.norm(x_integrated)), pad_len

class CausalBroadcaster(nn.Module):
    """Expands T/W semantic chunks back to T bytes CAUSALLY."""
    def __init__(self, config: OmegaConfig):
        super().__init__()
        self.W = config.decimation_rate
        # Trap A Fix: Learned Upsampling for gradient flow
        self.up_proj = nn.Linear(config.d_model, config.d_byte * self.W, bias=False)

    def forward(self, z_semantic, T_original, pad_len):
        B, N, D = z_semantic.shape
        
        # CRITICAL CAUSAL SHIFT: Shift everything right by 1.
        # Window N can only be used to predict bytes in Window N+1.
        z_shifted = torch.cat([torch.zeros_like(z_semantic[:, :1, :]), z_semantic[:, :-1, :]], dim=1)
        
        # Learned expansion
        z_expanded = self.up_proj(z_shifted) # [B, N, W * d_byte]
        z_broadcast = z_expanded.view(B, N * self.W, -1) # [B, N*W, d_byte]
        
        # Remove padding to match original byte sequence length
        if pad_len > 0:
            z_broadcast = z_broadcast[:, :-pad_len, :]
            
        return z_broadcast

class OmegaGDM(nn.Module):
    def __init__(self, config: OmegaConfig, force_sequential=False):
        super().__init__()
        self.config = config
        self.W = config.decimation_rate
        
        # 1. Byte-Catcher Manifold
        self.embedding = nn.Embedding(config.vocab_size, config.d_byte)
        catcher_cfg = HGDMConfig(
            d_model=config.d_byte, 
            n_layers=config.catcher_layers, 
            n_heads=4, 
            d_k=32, 
            d_v=32, 
            d_ff=config.d_byte*4
        )
        self.byte_catcher = nn.ModuleList([
            HGDMLayer(catcher_cfg, i, force_sequential) for i in range(config.catcher_layers)
        ])
        
        # 2. Phase Transition
        self.decimator = CausalDecimator(config)
        self.broadcaster = CausalBroadcaster(config)
        
        # Trap B Fix: Secondary Positional Embedding for the Semantic Core
        self.semantic_pos_embed = nn.Parameter(
            torch.randn(1, config.max_position_embeddings // self.W, config.d_model) * 0.02
        )
        
        # 3. Semantic Core Manifold (The Heavy Lifter)
        core_cfg = HGDMConfig(
            d_model=config.d_model, 
            n_layers=config.core_layers, 
            n_heads=config.n_heads, 
            d_k=config.d_k, 
            d_v=config.d_v, 
            d_ff=config.d_ff
        )
        self.semantic_core = nn.ModuleList([
            HGDMLayer(core_cfg, i, force_sequential) for i in range(config.core_layers)
        ])
        
        # 4. Byte-Renderer Manifold
        renderer_cfg = HGDMConfig(
            d_model=config.d_byte, 
            n_layers=config.renderer_layers, 
            n_heads=4, 
            d_k=32, 
            d_v=32, 
            d_ff=config.d_byte*4
        )
        self.byte_renderer = nn.ModuleList([
            HGDMLayer(renderer_cfg, i, force_sequential) for i in range(config.renderer_layers)
        ])
        
        self.norm_f = RMSNorm(config.d_byte)
        self.fc_out = nn.Linear(config.d_byte, config.vocab_size, bias=False)
        self.fc_out.weight = self.embedding.weight

    def forward(self, byte_seq, states=None, offset=0):
        B, T = byte_seq.shape
        x_byte = self.embedding(byte_seq)
        
        # Initialize states if None
        # Format: [catcher_states, core_states, renderer_states, buffer_state]
        if states is None:
            states = [
                [None] * self.config.catcher_layers, 
                [None] * self.config.core_layers, 
                [None] * self.config.renderer_layers,
                None # buffer_state
            ]
            
        next_states = [[], [], [], None]
        
        # Retrieve or initialize temporal buffer state for step-by-step generation
        buffer_state = states[3]
        if buffer_state is None:
            buffer_state = {
                'byte_buffer': torch.zeros(B, 0, self.config.d_byte, device=byte_seq.device, dtype=x_byte.dtype),
                'z_expanded_next': torch.zeros(B, self.W * self.config.d_byte, device=byte_seq.device, dtype=x_byte.dtype),
                'block_idx': offset // self.W
            }
            
        # ---------------------------------------------------------------------
        # CASE 1: PARALLEL PATH (T > 1) - Used during training and prompting
        # ---------------------------------------------------------------------
        if T > 1:
            # --- PHASE 1: BYTE-CATCHER ---
            for i, layer in enumerate(self.byte_catcher):
                x_byte, ns = layer(x_byte, states[0][i])
                next_states[0].append(ns)
                
            # --- PHASE 2: DECIMATION ---
            x_semantic, pad_len = self.decimator(x_byte)
            
            # Add positional embeddings causally with wrap-around support
            N = x_semantic.shape[1]
            semantic_offset = offset // self.W
            pos_offset = semantic_offset % self.semantic_pos_embed.shape[1]
            if pos_offset + N > self.semantic_pos_embed.shape[1]:
                indices = torch.arange(semantic_offset, semantic_offset + N, device=x_semantic.device) % self.semantic_pos_embed.shape[1]
                x_semantic = x_semantic + self.semantic_pos_embed[:, indices, :]
            else:
                x_semantic = x_semantic + self.semantic_pos_embed[:, pos_offset : pos_offset + N, :]
                
            # --- PHASE 3: SEMANTIC CORE ---
            for i, layer in enumerate(self.semantic_core):
                x_semantic, ns = layer(x_semantic, states[1][i])
                next_states[1].append(ns)
                
            # --- PHASE 4: BROADCAST & RENDER ---
            z_broadcast = self.broadcaster(x_semantic, T, pad_len)
            x_render = x_byte + z_broadcast  # Residual connection from catcher
            
            for i, layer in enumerate(self.byte_renderer):
                x_render, ns = layer(x_render, states[2][i])
                next_states[2].append(ns)
                
            # Update buffer state at sequence end for subsequent step-by-step decoding
            next_states[3] = {
                'byte_buffer': torch.zeros(B, 0, self.config.d_byte, device=byte_seq.device, dtype=x_byte.dtype),
                'z_expanded_next': self.broadcaster.up_proj(x_semantic[:, -1, :]),
                'block_idx': (offset + T) // self.W
            }
            
            x_out = self.norm_f(x_render)
            return self.fc_out(x_out), next_states

        # ---------------------------------------------------------------------
        # CASE 2: RECURRENT STEP PATH (T == 1) - Used during autoregressive generation
        # ---------------------------------------------------------------------
        else:
            # --- PHASE 1: BYTE-CATCHER ---
            x_byte_out = x_byte
            for i, layer in enumerate(self.byte_catcher):
                x_byte_out, ns = layer(x_byte_out, states[0][i])
                next_states[0].append(ns)
                
            # Accumulate current catcher output into the window buffer
            buffer_state['byte_buffer'] = torch.cat([buffer_state['byte_buffer'], x_byte_out], dim=1)
            i_win = buffer_state['byte_buffer'].shape[1] - 1
            
            # --- PHASE 2: CAUSAL BROADCAST EXTRACTION ---
            # Retrieve the specific slice of the pre-computed upsampled context for this step
            z_step = buffer_state['z_expanded_next'][:, i_win * self.config.d_byte : (i_win + 1) * self.config.d_byte].unsqueeze(1)
            x_render = x_byte_out + z_step
            
            # --- PHASE 3: BYTE-RENDER ---
            for i, layer in enumerate(self.byte_renderer):
                x_render, ns = layer(x_render, states[2][i])
                next_states[2].append(ns)
                
            # --- PHASE 4: CONDITIONAL DECIMATION & SEMANTIC CORE UPDATE ---
            if buffer_state['byte_buffer'].shape[1] == self.W:
                x_grouped = buffer_state['byte_buffer']
                buffer_state['byte_buffer'] = torch.zeros(B, 0, self.config.d_byte, device=byte_seq.device, dtype=x_byte.dtype)
                
                # Causal decay integration
                indices = torch.arange(self.W, device=byte_seq.device, dtype=torch.float32)
                causal_distances = (self.W - 1) - indices
                weights = torch.softmax(-F.softplus(self.decimator.lambda_decay) * causal_distances, dim=-1)
                
                x_integrated = (x_grouped * weights.view(1, self.W, 1)).sum(dim=1)
                x_semantic = self.decimator.proj(self.decimator.norm(x_integrated)).unsqueeze(1)
                
                # Add Semantic positional embedding
                block_idx = buffer_state['block_idx'] % self.semantic_pos_embed.shape[1]
                x_semantic = x_semantic + self.semantic_pos_embed[:, block_idx : block_idx + 1, :]
                
                # Step Semantic Core
                for i, layer in enumerate(self.semantic_core):
                    x_semantic, ns = layer(x_semantic, states[1][i])
                    next_states[1].append(ns)
                    
                # Prefetch next window's broadcasted upsampling projections
                buffer_state['z_expanded_next'] = self.broadcaster.up_proj(x_semantic).squeeze(1)
                buffer_state['block_idx'] += 1
            else:
                # Keep core states unchanged if semantic core did not step
                next_states[1] = states[1]
                
            next_states[3] = buffer_state
            
            x_out = self.norm_f(x_render)
            return self.fc_out(x_out), next_states

    @torch.no_grad()
    def generate(self, prompt_bytes, max_new_bytes=100, temp=0.8):
        self.eval()
        generated = prompt_bytes
        logits, states = self.forward(prompt_bytes)
        next_logit = logits[:, -1, :] / temp
        next_probs = F.softmax(next_logit, dim=-1)
        next_byte = torch.multinomial(next_probs, num_samples=1)
        generated = torch.cat([generated, next_byte], dim=1)
        
        offset = prompt_bytes.shape[1]
        for _ in range(max_new_bytes - 1):
            logits, next_states = self.forward(next_byte, states, offset=offset)
            states = next_states
            next_logit = logits[:, -1, :] / temp
            next_probs = F.softmax(next_logit, dim=-1)
            next_byte = torch.multinomial(next_probs, num_samples=1)
            generated = torch.cat([generated, next_byte], dim=1)
            offset += 1
        return generated
