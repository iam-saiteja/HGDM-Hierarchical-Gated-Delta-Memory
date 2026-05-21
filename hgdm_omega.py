import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from ultimate.hgdm_ultimate import HGDMLayer, RMSNorm, HGDMConfig, CrossLayerStateFusion

# =============================================================================
# OMEGAGDM V2 — Hierarchical Temporal Decimation (Bugfixed & Causal)
# =============================================================================

@dataclass
class OmegaConfig:
    d_byte: int = 256
    catcher_layers: int = 2
    renderer_layers: int = 2

    d_model: int = 2048
    core_layers: int = 20
    n_heads: int = 16
    d_k: int = 64
    d_v: int = 64
    d_ff: int = 8192

    decimation_rate: int = 8          # W = 8
    vocab_size: int = 256
    max_position_embeddings: int = 65536

    use_state_fusion: bool = False
    use_variable_delta_t: bool = False

class CausalBroadcaster(nn.Module):
    def __init__(self, config: OmegaConfig):
        super().__init__()
        self.W = config.decimation_rate
        self.proj = nn.Linear(config.d_model, config.d_byte, bias=False)

    def forward(self, z_semantic, T_original):
        B, N, _ = z_semantic.shape
        zeros = torch.zeros_like(z_semantic[:, :1, :])
        z_shifted = torch.cat([zeros, z_semantic], dim=1) 
        z_proj = self.proj(z_shifted)                  
        z_broadcast = z_proj.repeat_interleave(self.W, dim=1)  
        return z_broadcast[:, :T_original, :]

class OmegaGDM(nn.Module):
    def __init__(self, config: OmegaConfig, force_sequential=False):
        super().__init__()
        self.config = config
        self.W = config.decimation_rate

        self.embedding = nn.Embedding(config.vocab_size, config.d_byte)

        catcher_cfg = HGDMConfig(d_model=config.d_byte, n_layers=config.catcher_layers, n_heads=4, d_k=32, d_v=32, d_ff=config.d_byte * 4)
        self.byte_catcher = nn.ModuleList([HGDMLayer(catcher_cfg, i, force_sequential) for i in range(config.catcher_layers)])

        decimator_cfg = HGDMConfig(d_model=config.d_byte, n_layers=1, n_heads=4, d_k=32, d_v=32, d_ff=config.d_byte * 4)
        self.decimator_layer = HGDMLayer(decimator_cfg, 0, force_sequential)
        self.decimator_proj = nn.Linear(config.d_byte, config.d_model, bias=False)
        self.decimator_norm = RMSNorm(config.d_byte)

        self.semantic_pos_embed = nn.Parameter(torch.randn(1, config.max_position_embeddings // self.W, config.d_model) * 0.02)

        core_cfg = HGDMConfig(d_model=config.d_model, n_layers=config.core_layers, n_heads=config.n_heads, d_k=config.d_k, d_v=config.d_v, d_ff=config.d_ff)
        self.semantic_core = nn.ModuleList([HGDMLayer(core_cfg, i, force_sequential) for i in range(config.core_layers)])

        if config.use_state_fusion:
            self.core_state_fusion = CrossLayerStateFusion(core_cfg)

        H_core, dk_core, dv_core = config.n_heads, config.d_k, config.d_v
        H_ren, dk_ren, dv_ren = 4, 32, 32

        # Factored Bilinear State Highway
        self.highway_td_norm   = RMSNorm(dk_core * dv_core)
        self.highway_td_proj_k = nn.Linear(dk_core, dk_ren, bias=False)
        self.highway_td_proj_v = nn.Linear(dv_core, dv_ren, bias=False)
        self.highway_td_gate   = nn.Parameter(torch.full((H_ren,), -1.0))

        self.highway_bu_norm   = RMSNorm(dk_ren * dv_ren)
        self.highway_bu_proj_k = nn.Linear(dk_ren, dk_core, bias=False)
        self.highway_bu_proj_v = nn.Linear(dv_ren, dv_core, bias=False)
        self.highway_bu_gate   = nn.Parameter(torch.full((H_core,), -1.0))

        self._core_hw_shape = (H_core, dk_core, dv_core)
        self._ren_hw_shape  = (H_ren,  dk_ren,  dv_ren)

        self.broadcaster = CausalBroadcaster(config)

        renderer_cfg = HGDMConfig(d_model=config.d_byte, n_layers=config.renderer_layers, n_heads=4, d_k=32, d_v=32, d_ff=config.d_byte * 4)
        self.byte_renderer = nn.ModuleList([HGDMLayer(renderer_cfg, i, force_sequential) for i in range(config.renderer_layers)])

        self.norm_f  = RMSNorm(config.d_byte)
        self.fc_out  = nn.Linear(config.d_byte, config.vocab_size, bias=False)
        self.fc_out.weight = self.embedding.weight

    def _apply_td_highway(self, S_core_last, renderer_state_0):
        B = S_core_last.shape[0]
        S_mean = S_core_last.mean(dim=1)
        S_mean = self.highway_td_norm(S_mean.reshape(B, -1)).reshape(B, self._core_hw_shape[1], self._core_hw_shape[2])
        S_proj = self.highway_td_proj_k(S_mean.transpose(-1, -2)).transpose(-1, -2)
        S_proj = self.highway_td_proj_v(S_proj).unsqueeze(1).expand(-1, self._ren_hw_shape[0], -1, -1)
        gate = torch.sigmoid(self.highway_td_gate)[None, :, None, None]
        if renderer_state_0 is None: return gate * S_proj
        return renderer_state_0 + gate * S_proj

    def _apply_bu_highway(self, S_renderer_last, core_state_0):
        if S_renderer_last is None: return core_state_0
        B = S_renderer_last.shape[0]
        S_mean = S_renderer_last.mean(dim=1)
        S_mean = self.highway_bu_norm(S_mean.reshape(B, -1)).reshape(B, self._ren_hw_shape[1], self._ren_hw_shape[2])
        S_proj = self.highway_bu_proj_k(S_mean.transpose(-1, -2)).transpose(-1, -2)
        S_proj = self.highway_bu_proj_v(S_proj).unsqueeze(1).expand(-1, self._core_hw_shape[0], -1, -1)
        gate = torch.sigmoid(self.highway_bu_gate)[None, :, None, None]
        if core_state_0 is None: return gate * S_proj
        return core_state_0 + gate * S_proj

    def forward(self, byte_seq, states=None, offset=0):
        B, T = byte_seq.shape
        x_byte = self.embedding(byte_seq)

        if states is None:
            states = [[None]*self.config.catcher_layers, [None]*self.config.core_layers, [None]*self.config.renderer_layers, None, None]
        next_states = [[], [], [], None, None]

        # ==========================================
        # TRAINING BLOCK (T > 1) - Chunkwise Unrolled
        # ==========================================
        if T > 1:
            # 1. Catcher & Decimator run in pure parallel
            for i, layer in enumerate(self.byte_catcher):
                x_byte, ns = layer(x_byte, states[0][i])
                next_states[0].append(ns)

            x_dec, dec_ns = self.decimator_layer(x_byte, states[3])
            next_states[3] = dec_ns

            N = T // self.W
            # --- Edge Case: T < W (prompt shorter than a single chunk) ---
            if N == 0:
                x_render = x_byte  # No broadcast yet, core hasn't fired
                ren_states = list(states[2])
                for i, layer in enumerate(self.byte_renderer):
                    x_render, ns = layer(x_render, ren_states[i])
                    ren_states[i] = ns

                next_states[1] = list(states[1])  # Core states unchanged
                next_states[2] = ren_states
                next_states[4] = {
                    'byte_pos': T % self.W,
                    'z_broadcast_cache': torch.zeros(B, self.config.d_byte, device=x_byte.device, dtype=x_byte.dtype),
                    'prev_renderer_last_S': ren_states[-1] if ren_states[-1] is not None else None,
                }
                x_out = self.norm_f(x_render)
                return self.fc_out(x_out), next_states

            # --- Main Path: Chunkwise Sequential Loop (N >= 1) ---
            x_render_out = []
            core_states = list(states[1])
            ren_states = list(states[2])

            buf = states[4] or {}
            prev_raw_ren_ns = buf.get('prev_renderer_last_S', None)
            z_bc_cache = None

            for chunk_idx in range(N):
                # --- CORE PHASE (1 semantic step per W-byte chunk) ---
                x_sem_chunk = x_dec[:, (chunk_idx + 1) * self.W - 1 : (chunk_idx + 1) * self.W, :]
                x_sem_chunk = self.decimator_proj(self.decimator_norm(x_sem_chunk))

                semantic_offset = offset // self.W + chunk_idx
                pos_idx = semantic_offset % self.semantic_pos_embed.shape[1]
                x_sem_chunk = x_sem_chunk + self.semantic_pos_embed[:, pos_idx : pos_idx + 1, :]

                # Bottom-Up Highway: previous chunk's Renderer state -> Core layer 0
                core_states[0] = self._apply_bu_highway(prev_raw_ren_ns, core_states[0])

                x_semantic = x_sem_chunk
                prev_raw_ns = None
                for i, layer in enumerate(self.semantic_core):
                    x_semantic, ns = layer(x_semantic, core_states[i])
                    raw_ns = ns
                    if self.config.use_state_fusion and i > 0 and prev_raw_ns is not None:
                        ns = self.core_state_fusion.fuse(ns, prev_raw_ns, i)
                    prev_raw_ns = raw_ns
                    core_states[i] = ns

                S_core_last = raw_ns
                # Broadcast projection (differentiable in training, no detach)
                z_broadcast = self.broadcaster.proj(x_semantic).repeat_interleave(self.W, dim=1)
                if chunk_idx == N - 1:
                    z_bc_cache = self.broadcaster.proj(x_semantic).detach()

                # --- RENDERER PHASE (W bytes) ---
                # Top-Down Highway: Core's final state -> Renderer layer 0
                ren_states[0] = self._apply_td_highway(S_core_last, ren_states[0])

                byte_chunk = x_byte[:, chunk_idx * self.W : (chunk_idx + 1) * self.W, :]
                x_render = byte_chunk + z_broadcast

                for i, layer in enumerate(self.byte_renderer):
                    x_render, ns = layer(x_render, ren_states[i])
                    ren_states[i] = ns
                    if i == self.config.renderer_layers - 1:
                        prev_raw_ren_ns = ns

                x_render_out.append(x_render)

            # Handle remainder bytes if T is not perfectly divisible by W
            remainder = T % self.W
            if remainder > 0:
                byte_chunk = x_byte[:, N * self.W :, :]
                x_render_rem = byte_chunk + z_bc_cache.repeat_interleave(remainder, dim=1)
                for i, layer in enumerate(self.byte_renderer):
                    x_render_rem, ns = layer(x_render_rem, ren_states[i])
                    ren_states[i] = ns
                    if i == self.config.renderer_layers - 1:
                        prev_raw_ren_ns = ns
                x_render_out.append(x_render_rem)

            x_render_full = torch.cat(x_render_out, dim=1)

            next_states[1] = core_states
            next_states[2] = ren_states
            next_states[4] = {
                'byte_pos': remainder,
                'z_broadcast_cache': z_bc_cache.squeeze(1) if z_bc_cache is not None else torch.zeros(B, self.config.d_byte, device=x_byte.device, dtype=x_byte.dtype),
                'prev_renderer_last_S': prev_raw_ren_ns,
            }

            x_out = self.norm_f(x_render_full)
            return self.fc_out(x_out), next_states

        # ==========================================
        # GENERATION BLOCK (T == 1)
        # ==========================================
        else:
            for i, layer in enumerate(self.byte_catcher):
                x_byte, ns = layer(x_byte, states[0][i])
                next_states[0].append(ns)

            x_dec_step, dec_ns = self.decimator_layer(x_byte, states[3])
            next_states[3] = dec_ns

            buf = states[4]
            if buf is None:
                buf = {'byte_pos': 0, 'z_broadcast_cache': torch.zeros(B, self.config.d_byte, device=byte_seq.device, dtype=x_byte.dtype), 'prev_renderer_last_S': None}

            byte_pos = buf['byte_pos']
            z_bc = buf['z_broadcast_cache']
            x_render = x_byte + z_bc.unsqueeze(1)

            renderer_states_in = list(states[2])
            raw_ren_ns = None
            for i, layer in enumerate(self.byte_renderer):
                x_render, ns = layer(x_render, renderer_states_in[i])
                raw_ren_ns = ns
                next_states[2].append(ns)

            next_byte_pos = byte_pos + 1
            new_buf = {'byte_pos': next_byte_pos % self.W, 'z_broadcast_cache': z_bc, 'prev_renderer_last_S': raw_ren_ns}

            if next_byte_pos == self.W:
                x_sem_chunk = self.decimator_proj(self.decimator_norm(x_dec_step))
                
                chunk_idx = offset // self.W 
                pos_i = chunk_idx % self.semantic_pos_embed.shape[1]
                x_sem_chunk = x_sem_chunk + self.semantic_pos_embed[:, pos_i : pos_i + 1, :]

                core_init_0 = self._apply_bu_highway(raw_ren_ns, states[1][0])
                core_states_in = [core_init_0] + [states[1][i] for i in range(1, self.config.core_layers)]

                x_semantic = x_sem_chunk
                prev_raw_ns = None
                for i, layer in enumerate(self.semantic_core):
                    x_semantic, ns = layer(x_semantic, core_states_in[i])
                    raw_ns = ns
                    if self.config.use_state_fusion and i > 0 and prev_raw_ns is not None:
                        ns = self.core_state_fusion.fuse(ns, prev_raw_ns, i)
                    prev_raw_ns = raw_ns
                    next_states[1].append(ns)

                S_core_last = raw_ns
                new_buf['z_broadcast_cache'] = self.broadcaster.proj(x_semantic[:, 0, :]).detach()
                next_states[2][0] = self._apply_td_highway(S_core_last, next_states[2][0])
            else:
                next_states[1] = list(states[1])

            next_states[4] = new_buf
            x_out = self.norm_f(x_render)
            return self.fc_out(x_out), next_states

    @torch.no_grad()
    def generate(self, prompt_bytes, max_new_bytes=100, temp=0.8):
        self.eval()
        generated = prompt_bytes
        logits, states = self.forward(prompt_bytes)
        next_logit = logits[:, -1, :] / temp
        next_byte = torch.multinomial(F.softmax(next_logit, dim=-1), num_samples=1)
        generated = torch.cat([generated, next_byte], dim=1)

        offset = prompt_bytes.shape[1]
        for _ in range(max_new_bytes - 1):
            logits, states = self.forward(next_byte, states, offset=offset)
            next_logit = logits[:, -1, :] / temp
            next_byte = torch.multinomial(F.softmax(next_logit, dim=-1), num_samples=1)
            generated = torch.cat([generated, next_byte], dim=1)
            offset += 1
        return generated