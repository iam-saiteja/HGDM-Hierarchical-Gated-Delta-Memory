import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# Import with fallback for portability
try:
    from ultimate.hgdm_ultimate import HGDMLayer, RMSNorm, HGDMConfig, CrossLayerStateFusion
except (ImportError, ModuleNotFoundError):
    try:
        from hgdm_ultimate import HGDMLayer, RMSNorm, HGDMConfig, CrossLayerStateFusion
    except (ImportError, ModuleNotFoundError, ValueError):
        try:
            from .ultimate.hgdm_ultimate import HGDMLayer, RMSNorm, HGDMConfig, CrossLayerStateFusion
        except (ImportError, ModuleNotFoundError, ValueError):
            try:
                from .hgdm_ultimate import HGDMLayer, RMSNorm, HGDMConfig, CrossLayerStateFusion
            except (ImportError, ModuleNotFoundError, ValueError) as exc:
                raise ImportError(
                    "Could not import HGDMLayer from ultimate.hgdm_ultimate or hgdm_ultimate. "
                    "Ensure ultimate/ is in your Python path or run from project root."
                ) from exc

# =============================================================================
# OMEGAGDM V2 — Hierarchical Temporal Decimation (Bugfixed)
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
    use_variable_delta_t: bool = True   # [STEP-01] Time-based model: content-driven Δt decay (CfC/Mamba)

class CausalBroadcaster(nn.Module):
    def __init__(self, config: OmegaConfig):
        super().__init__()
        self.W = config.decimation_rate
        self.proj = nn.Linear(config.d_model, config.d_byte, bias=False)

    def forward(self, z_semantic, T_original):
        zeros = torch.zeros_like(z_semantic[:, :1, :])
        # Shift by one chunk so each byte only sees prior semantic context.
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

        catcher_cfg = HGDMConfig(d_model=config.d_byte, n_layers=config.catcher_layers, n_heads=4, d_k=32, d_v=32, d_ff=config.d_byte * 4, use_variable_delta_t=config.use_variable_delta_t)
        self.byte_catcher = nn.ModuleList([HGDMLayer(catcher_cfg, i, force_sequential) for i in range(config.catcher_layers)])

        decimator_cfg = HGDMConfig(d_model=config.d_byte, n_layers=1, n_heads=4, d_k=32, d_v=32, d_ff=config.d_byte * 4, use_variable_delta_t=config.use_variable_delta_t)
        self.decimator_layer = HGDMLayer(decimator_cfg, 0, force_sequential)
        self.decimator_proj = nn.Linear(config.d_byte, config.d_model, bias=False)
        self.decimator_norm = RMSNorm(config.d_byte)

        # [STEP-11] Absolute positional embedding removed (saves VRAM, using RoPE in core mixer instead)
        # self.semantic_pos_embed removed

        core_cfg = HGDMConfig(
            d_model=config.d_model, 
            n_layers=config.core_layers, 
            n_heads=config.n_heads, 
            d_k=config.d_k, 
            d_v=config.d_v, 
            d_ff=config.d_ff,
            use_rope=True,
            max_position_embeddings=config.max_position_embeddings // self.W
        )
        self.semantic_core = nn.ModuleList([HGDMLayer(core_cfg, i, force_sequential) for i in range(config.core_layers)])

        if config.use_state_fusion:
            self.core_state_fusion = CrossLayerStateFusion(core_cfg)

        H_core, dk_core, dv_core = config.n_heads, config.d_k, config.d_v
        H_ren, dk_ren, dv_ren = 4, 32, 32

        # Factored Bilinear State Highway
        self.highway_td_norm   = RMSNorm(dk_core * dv_core)
        self.highway_td_proj_k = nn.Linear(dk_core, dk_ren, bias=False)
        self.highway_td_proj_v = nn.Linear(dv_core, dv_ren, bias=False)
        # [STEP-13] Content-dependent TD highway gate network
        self.td_gate_net = nn.Linear(config.d_model, H_ren, bias=True)
        with torch.no_grad():
            self.td_gate_net.weight.zero_()
            self.td_gate_net.bias.fill_(-4.0)

        self.highway_bu_norm   = RMSNorm(dk_ren * dv_ren)
        self.highway_bu_proj_k = nn.Linear(dk_ren, dk_core, bias=False)
        self.highway_bu_proj_v = nn.Linear(dv_ren, dv_core, bias=False)
        # [STEP-13] Content-dependent BU highway gate network
        self.bu_gate_net = nn.Linear(config.d_byte, H_core, bias=True)
        with torch.no_grad():
            self.bu_gate_net.weight.zero_()
            self.bu_gate_net.bias.fill_(-2.0)

        self._core_hw_shape = (H_core, dk_core, dv_core)
        self._ren_hw_shape  = (H_ren,  dk_ren,  dv_ren)

        self.broadcaster = CausalBroadcaster(config)

        renderer_cfg = HGDMConfig(d_model=config.d_byte, n_layers=config.renderer_layers, n_heads=4, d_k=32, d_v=32, d_ff=config.d_byte * 4, use_variable_delta_t=config.use_variable_delta_t)
        self.byte_renderer = nn.ModuleList([HGDMLayer(renderer_cfg, i, force_sequential) for i in range(config.renderer_layers)])

        self.norm_f  = RMSNorm(config.d_byte)
        self.fc_out  = nn.Linear(config.d_byte, config.vocab_size, bias=False)
        self.fc_out.weight = self.embedding.weight

        # [STEP-12] Content-Aware Decimation: boundary head initialized with bias=-2.08 (p≈0.11≈1/8)
        self.boundary_head = nn.Linear(config.d_byte, 1, bias=True)
        with torch.no_grad():
            self.boundary_head.weight.zero_()
            self.boundary_head.bias.fill_(-2.08)

        # [STEP-14] Multi-Token Prediction K=4 heads (sharing embedding weights)
        self.mtp_heads = nn.ModuleList([nn.Linear(config.d_byte, config.vocab_size, bias=False) for _ in range(3)])
        for head in self.mtp_heads:
            head.weight = self.embedding.weight

    def _decimation_step(self, cum_prob, byte_pos, boundary_prob):
        new_cum_prob = cum_prob + boundary_prob
        new_byte_pos = byte_pos + 1
        threshold_trigger = new_cum_prob >= 1.0
        window_trigger = new_byte_pos >= self.W
        trigger_mask = threshold_trigger | window_trigger
        final_cum_prob = torch.where(
            trigger_mask,
            torch.where(threshold_trigger, new_cum_prob - 1.0, torch.zeros_like(new_cum_prob)),
            new_cum_prob,
        )
        final_byte_pos = torch.where(trigger_mask, torch.zeros_like(new_byte_pos), new_byte_pos)
        return trigger_mask, final_cum_prob, final_byte_pos

    def _online_decimation_events(self, boundary_prob, states, B, device):
        buf = states[4] if states is not None else None
        if buf is not None and 'cum_prob' in buf:
            cum_prob = buf['cum_prob'].to(device=device, dtype=boundary_prob.dtype)
        else:
            cum_prob = torch.zeros(B, device=device, dtype=boundary_prob.dtype)

        if buf is not None and 'byte_pos' in buf:
            byte_pos = buf['byte_pos'].to(device=device, dtype=torch.long)
        else:
            byte_pos = torch.zeros(B, device=device, dtype=torch.long)

        events = []
        for t in range(boundary_prob.shape[1]):
            trigger_mask, cum_prob, byte_pos = self._decimation_step(cum_prob, byte_pos, boundary_prob[:, t])
            events.append(trigger_mask)

        return torch.stack(events, dim=1), cum_prob, byte_pos

    def _apply_td_highway(self, S_core_last, renderer_state_0, x_core=None):
        # [STEP-05] State is now (S, n) tuple — extract S matrix for highway projection
        S_core = S_core_last[0] if isinstance(S_core_last, tuple) else S_core_last
        ren_S  = renderer_state_0[0] if isinstance(renderer_state_0, tuple) else renderer_state_0
        ren_n  = renderer_state_0[1] if isinstance(renderer_state_0, tuple) else None
        
        if S_core is None:
            return (ren_S, ren_n)
            
        B = S_core.shape[0]
        S_mean = S_core.mean(dim=1)
        S_mean = self.highway_td_norm(S_mean.reshape(B, -1)).reshape(B, self._core_hw_shape[1], self._core_hw_shape[2])
        S_proj = self.highway_td_proj_k(S_mean.transpose(-1, -2)).transpose(-1, -2)
        S_proj = self.highway_td_proj_v(S_proj).unsqueeze(1).expand(-1, self._ren_hw_shape[0], -1, -1)
        
        if x_core is not None:
            gate = torch.sigmoid(self.td_gate_net(x_core))[:, :, None, None]
        else:
            gate = torch.sigmoid(self.td_gate_net.bias)[None, :, None, None]
            
        new_S = gate * S_proj if ren_S is None else ren_S + gate * S_proj
        return (new_S, ren_n)  # return (S, n) tuple to keep state format consistent

    def _apply_bu_highway(self, S_renderer_last, core_state_0, x_dec_last=None):
        # [STEP-05] State is now (S, n) tuple — extract S matrix for highway projection
        S_ren  = S_renderer_last[0] if isinstance(S_renderer_last, tuple) else S_renderer_last
        core_S = core_state_0[0] if isinstance(core_state_0, tuple) else core_state_0
        core_n = core_state_0[1] if isinstance(core_state_0, tuple) else None
        
        if S_ren is None:
            return (core_S, core_n)
            
        B = S_ren.shape[0]
        S_mean = S_ren.mean(dim=1)
        S_mean = self.highway_bu_norm(S_mean.reshape(B, -1)).reshape(B, self._ren_hw_shape[1], self._ren_hw_shape[2])
        S_proj = self.highway_bu_proj_k(S_mean.transpose(-1, -2)).transpose(-1, -2)
        S_proj = self.highway_bu_proj_v(S_proj).unsqueeze(1).expand(-1, self._core_hw_shape[0], -1, -1)
        
        if x_dec_last is not None:
            gate = torch.sigmoid(self.bu_gate_net(x_dec_last))[:, :, None, None]
        else:
            gate = torch.sigmoid(self.bu_gate_net.bias)[None, :, None, None]
            
        new_S = gate * S_proj if core_S is None else core_S + gate * S_proj
        return (new_S, core_n)  # return (S, n) tuple to keep state format consistent

    def forward(self, byte_seq=None, states=None, offset=0, return_mtp=False, x_embed=None, return_latent=False, return_boundary_loss=False, return_states=True):
        if byte_seq is not None:
            B, T = byte_seq.shape
            x_byte = self.embedding(byte_seq)
        elif x_embed is not None:
            B, T, _ = x_embed.shape
            x_byte = x_embed
        else:
            raise ValueError("Either byte_seq or x_embed must be provided")

        if states is None:
            states = [[None]*self.config.catcher_layers, [None]*self.config.core_layers, [None]*self.config.renderer_layers, None, None]
        next_states = [[], [], [], None, None]

        if T > 1:
            for i, layer in enumerate(self.byte_catcher):
                x_byte, ns = layer(x_byte, states[0][i])
                next_states[0].append(ns)

            x_dec, dec_ns = self.decimator_layer(x_byte, states[3])
            next_states[3] = dec_ns

            # BUG 1 FIX: Only slice completed windows, no F.pad poisoning
            N = T // self.W
            final_cum_prob = None
            final_byte_pos = None
            b_loss = None
            if N > 0:
                # [STEP-12] Content-Aware Decimation
                boundary_logit = self.boundary_head(x_byte) # [B, T, 1]
                boundary_prob = torch.sigmoid(boundary_logit).squeeze(-1) # [B, T]

                decimation_events, final_cum_prob, final_byte_pos = self._online_decimation_events(
                    boundary_prob, states, B, x_byte.device
                )

                # FULL-SEQUENCE DECIMATION POLICY:
                # Match streaming's online cumulative trigger rule and keep the fixed training
                # budget by taking the first N completed semantic events.
                positions = torch.arange(T, device=x_byte.device).unsqueeze(0).expand(B, -1)
                score = torch.where(
                    decimation_events,
                    (T - positions).to(boundary_prob.dtype),
                    torch.full_like(boundary_prob, -1.0),
                )
                _, top_idx = torch.topk(score, k=N, dim=-1)
                selected_valid = torch.gather(decimation_events, 1, top_idx)
                fallback_indices = torch.arange(N, device=x_byte.device).unsqueeze(0).expand(B, -1)
                fallback_indices = ((fallback_indices + 1) * self.W - 1).clamp_max(T - 1)
                top_idx = torch.where(selected_valid, top_idx, fallback_indices)
                selected_indices, _ = torch.sort(top_idx, dim=-1) # [B, N]

                boundary_prob_selected = torch.gather(boundary_prob, 1, selected_indices) # [B, N]
                x_semantic_in = torch.gather(x_dec, 1, selected_indices.unsqueeze(-1).expand(-1, -1, x_dec.shape[-1]))
                x_semantic_in = x_semantic_in * boundary_prob_selected.unsqueeze(-1)

                if return_boundary_loss:
                    b_loss = (boundary_prob * (1.0 - boundary_prob)).mean()
                x_semantic_in = self.decimator_proj(self.decimator_norm(x_semantic_in))

                semantic_offset = offset // self.W
                # [STEP-11] Absolute positional embedding removed (RoPE is applied inside the core's MultiHeadGatedDelta instead)
                
                core_init_0 = states[1][0]
                if states[4] is not None and states[4].get('prev_renderer_last_S') is not None:
                    core_init_0 = self._apply_bu_highway(states[4]['prev_renderer_last_S'], core_init_0, states[4].get('x_dec_last'))

                core_states_in = [core_init_0] + [states[1][i] for i in range(1, self.config.core_layers)]
                x_semantic = x_semantic_in
                prev_raw_ns = None
                for i, layer in enumerate(self.semantic_core):
                    x_semantic, ns = layer(x_semantic, core_states_in[i], offset=semantic_offset)
                    raw_ns = ns
                    if self.config.use_state_fusion and i > 0 and prev_raw_ns is not None:
                        ns = self.core_state_fusion.fuse(ns, prev_raw_ns, i)
                    prev_raw_ns = raw_ns
                    next_states[1].append(ns)

                S_core_last = raw_ns
                z_broadcast = self.broadcaster(x_semantic, T)
                z_broadcast_cache = self.broadcaster.proj(x_semantic[:, -1, :]).detach()
                renderer_init_0 = self._apply_td_highway(S_core_last, states[2][0], x_semantic[:, -1, :])
            else:
                z_broadcast = torch.zeros(B, T, self.config.d_byte, device=x_byte.device, dtype=x_byte.dtype)
                if states[4] is not None:
                    z_broadcast_cache = states[4]['z_broadcast_cache']
                else:
                    z_broadcast_cache = torch.zeros(B, self.config.d_byte, device=x_byte.device, dtype=x_byte.dtype)
                renderer_init_0 = states[2][0]
                next_states[1] = list(states[1])

            x_render = x_byte + z_broadcast
            renderer_states_in = [renderer_init_0] + [states[2][i] for i in range(1, self.config.renderer_layers)]

            raw_ren_ns = None
            for i, layer in enumerate(self.byte_renderer):
                x_render, ns = layer(x_render, renderer_states_in[i])
                raw_ren_ns = ns
                next_states[2].append(ns)

            if return_states:
                next_states[4] = {
                    'z_broadcast_cache': z_broadcast_cache,
                    'prev_renderer_last_S': raw_ren_ns,
                    'x_dec_last': x_dec[:, -1, :].detach(),
                    'cum_prob': final_cum_prob if final_cum_prob is not None else (
                        states[4]['cum_prob'] if states[4] is not None and 'cum_prob' in states[4]
                        else torch.zeros(B, device=x_byte.device, dtype=x_byte.dtype)
                    ),
                    'byte_pos': final_byte_pos if final_byte_pos is not None else (
                        states[4]['byte_pos'] if states[4] is not None and 'byte_pos' in states[4]
                        else torch.zeros(B, device=x_byte.device, dtype=torch.long)
                    )
                }
            else:
                next_states = None

            x_out = self.norm_f(x_render)
            logits1 = self.fc_out(x_out)
            if return_mtp:
                logits_all = [logits1] + [head(x_out) for head in self.mtp_heads]
                out_logits = logits_all
            else:
                out_logits = logits1
                
            if return_boundary_loss:
                if return_latent:
                    return out_logits, next_states, x_out, b_loss
                else:
                    return out_logits, next_states, b_loss
            else:
                if return_latent:
                    return out_logits, next_states, x_out
                else:
                    return out_logits, next_states

        else:
            for i, layer in enumerate(self.byte_catcher):
                x_byte, ns = layer(x_byte, states[0][i])
                next_states[0].append(ns)

            x_dec_step, dec_ns = self.decimator_layer(x_byte, states[3])
            next_states[3] = dec_ns

            buf = states[4]
            if buf is None:
                buf = {
                    'cum_prob': torch.zeros(B, device=x_byte.device), 
                    'byte_pos': torch.zeros(B, device=x_byte.device, dtype=torch.long),
                    'z_broadcast_cache': torch.zeros(B, self.config.d_byte, device=x_byte.device, dtype=x_byte.dtype), 
                    'prev_renderer_last_S': None
                }
            elif 'cum_prob' not in buf or 'byte_pos' not in buf:
                buf = {
                    'cum_prob': buf.get('cum_prob', torch.zeros(B, device=x_byte.device)),
                    'byte_pos': buf.get('byte_pos', torch.zeros(B, device=x_byte.device, dtype=torch.long)),
                    'z_broadcast_cache': buf['z_broadcast_cache'],
                    'prev_renderer_last_S': buf['prev_renderer_last_S'],
                    'x_dec_last': buf.get('x_dec_last')
                }

            cum_prob = buf['cum_prob']
            byte_pos = buf['byte_pos']
            z_bc = buf['z_broadcast_cache']
            x_render = x_byte + z_bc.unsqueeze(1)

            renderer_states_in = list(states[2])
            raw_ren_ns = None
            for i, layer in enumerate(self.byte_renderer):
                x_render, ns = layer(x_render, renderer_states_in[i])
                raw_ren_ns = ns
                next_states[2].append(ns)

            # STREAMING DECIMATION POLICY:
            # Per-sample online trigger shared with full-sequence training: emit when cumulative
            # boundary probability crosses 1.0, or when a decimation window completes.
            boundary_logit = self.boundary_head(x_byte) # [B, 1, 1]
            boundary_prob = torch.sigmoid(boundary_logit).squeeze(-1).squeeze(-1) # [B]
            trigger_mask, final_cum_prob, final_byte_pos = self._decimation_step(cum_prob, byte_pos, boundary_prob)

            new_buf = {
                'cum_prob': final_cum_prob,
                'byte_pos': final_byte_pos,
                'z_broadcast_cache': z_bc,
                'prev_renderer_last_S': raw_ren_ns,
                'x_dec_last': x_dec_step.squeeze(1).detach()
            }

            if trigger_mask.any().item():
                # Compact-trigger semantic core: run only on rows that emitted a token.
                
                x_sem_chunk = self.decimator_proj(self.decimator_norm(x_dec_step))
                x_sem_chunk = x_sem_chunk * boundary_prob.unsqueeze(1).unsqueeze(2)
                
                # Extract indices of triggered samples.
                triggered_indices = torch.nonzero(trigger_mask, as_tuple=True)[0]
                # [STEP-11] Absolute positional embedding removed (RoPE is applied inside the core's MultiHeadGatedDelta instead)
                chunk_idx = offset // self.W 
                
                # Only compute highway for triggered samples
                raw_ren_ns_triggered = tuple(t[triggered_indices] for t in raw_ren_ns) if isinstance(raw_ren_ns, tuple) else raw_ren_ns[triggered_indices]
                core_init_0_triggered = self._apply_bu_highway(
                    raw_ren_ns_triggered, 
                    (states[1][0][0][triggered_indices], states[1][0][1][triggered_indices]) if states[1][0] is not None else None,
                    x_dec_step[triggered_indices].squeeze(1)
                )
                
                # Gather triggered core states
                core_states_in = [core_init_0_triggered]
                for i in range(1, self.config.core_layers):
                    if states[1][i] is not None:
                        core_states_in.append((states[1][i][0][triggered_indices], states[1][i][1][triggered_indices]))
                    else:
                        core_states_in.append(None)
                
                # Process semantic core only for triggered samples
                x_semantic = x_sem_chunk[triggered_indices]
                prev_raw_ns = None
                
                for i, layer in enumerate(self.semantic_core):
                    x_semantic, ns = layer(x_semantic, core_states_in[i], offset=chunk_idx)
                    raw_ns = ns
                    if self.config.use_state_fusion and i > 0 and prev_raw_ns is not None:
                        ns = self.core_state_fusion.fuse(ns, prev_raw_ns, i)
                    prev_raw_ns = raw_ns
                    
                    # FIX: Guard against None states on cold-start streaming
                    if states[1][i] is not None:
                        old_S, old_n = states[1][i][0][triggered_indices], states[1][i][1][triggered_indices]
                    else:
                        old_S, old_n = None, None
                    
                    new_S, new_n = ns
                    if old_S is not None and old_n is not None:
                        final_S = new_S  # Already only triggered samples
                        final_n = new_n
                    else:
                        final_S, final_n = new_S, new_n
                    
                    # Scatter back to full batch
                    full_S = states[1][i][0] if states[1][i] is not None else None
                    full_n = states[1][i][1] if states[1][i] is not None else None
                    
                    if full_S is not None and full_n is not None:
                        final_S_full = full_S.clone()
                        final_n_full = full_n.clone()
                        final_S_full[triggered_indices] = final_S
                        final_n_full[triggered_indices] = final_n
                        next_states[1].append((final_S_full, final_n_full))
                    else:
                        # Cold start: create full batch states
                        final_S_full = torch.zeros(B, *final_S.shape[1:], dtype=final_S.dtype, device=final_S.device)
                        final_n_full = torch.zeros(B, *final_n.shape[1:], dtype=final_n.dtype, device=final_n.device)
                        final_S_full[triggered_indices] = final_S
                        final_n_full[triggered_indices] = final_n
                        next_states[1].append((final_S_full, final_n_full))

                S_core_last = raw_ns
                new_z_bc = self.broadcaster.proj(x_semantic[:, 0, :]).detach()
                
                # Scatter broadcaster output back to full batch
                z_bc_full = z_bc.clone() if z_bc is not None else torch.zeros_like(x_dec_step[:, 0, :])
                z_bc_full[triggered_indices] = new_z_bc
                new_buf['z_broadcast_cache'] = z_bc_full
                
                # Update TD highway (layer 0 renderer state) for triggered samples only
                if next_states[2][0] is not None:
                    old_render_S0_S, old_render_S0_n = next_states[2][0][0][triggered_indices], next_states[2][0][1][triggered_indices]
                else:
                    old_render_S0_S, old_render_S0_n = None, None
                
                new_render_S0 = self._apply_td_highway(
                    S_core_last,
                    (old_render_S0_S, old_render_S0_n) if old_render_S0_S is not None else None,
                    x_semantic[:, 0, :]
                )
                new_render_S0_S, new_render_S0_n = new_render_S0

                if next_states[2][0] is not None:
                    # Scatter triggered samples back to full batch
                    final_render_S0_S = next_states[2][0][0].clone()
                    final_render_S0_n = next_states[2][0][1].clone()
                    final_render_S0_S[triggered_indices] = new_render_S0_S
                    final_render_S0_n[triggered_indices] = new_render_S0_n
                else:
                    # Cold start: create full batch states
                    final_render_S0_S = torch.zeros(B, *new_render_S0_S.shape[1:], dtype=new_render_S0_S.dtype, device=new_render_S0_S.device)
                    final_render_S0_n = torch.zeros(B, *new_render_S0_n.shape[1:], dtype=new_render_S0_n.dtype, device=new_render_S0_n.device)
                    final_render_S0_S[triggered_indices] = new_render_S0_S
                    final_render_S0_n[triggered_indices] = new_render_S0_n
                next_states[2][0] = (final_render_S0_S, final_render_S0_n)
            else:
                next_states[1] = list(states[1])

            next_states[4] = new_buf
            x_out = self.norm_f(x_render)
            logits1 = self.fc_out(x_out)
            if return_mtp:
                logits_all = [logits1] + [head(x_out) for head in self.mtp_heads]
                out_logits = logits_all
            else:
                out_logits = logits1
                
            if return_latent:
                return out_logits, next_states, x_out
            else:
                return out_logits, next_states

    @torch.no_grad()
    def generate(self, prompt_bytes, max_new_bytes=100, temp=0.8, think_steps=0):
        if max_new_bytes == 0:
            return prompt_bytes
        self.eval()
        generated = [prompt_bytes]
        logits, states = self.forward(prompt_bytes)
        next_logit = logits[:, -1, :] / temp
        next_byte = torch.multinomial(F.softmax(next_logit, dim=-1), num_samples=1)
        generated.append(next_byte)

        offset = prompt_bytes.shape[1]
        for _ in range(max_new_bytes - 1):
            if think_steps > 0:
                states, _ = latent_think(self, states, n_thoughts=think_steps, temp=temp, offset=offset)
                offset += think_steps
            logits, states = self.forward(next_byte, states, offset=offset)
            next_logit = logits[:, -1, :] / temp
            next_byte = torch.multinomial(F.softmax(next_logit, dim=-1), num_samples=1)
            generated.append(next_byte)
            offset += 1
        return torch.cat(generated, dim=1)

def latent_think(model, states, n_thoughts=8, temp=0.3, offset=0):
    device = next(model.parameters()).device
    B = 1
    if states is not None:
        for state_group in states[:3]:
            if state_group is None:
                continue
            for state in state_group:
                if state is None:
                    continue
                B = state[0].shape[0] if isinstance(state, tuple) else state.shape[0]
                break
            if B != 1:
                break
    
    # Initialize thinking using embedding of a dummy zero byte
    x_byte = model.embedding(torch.zeros(B, 1, dtype=torch.long, device=device))
    
    current_states = states
    x_out_curr = x_byte
    
    thought_tokens = []
    
    for _ in range(n_thoughts):
        logits, current_states, x_out_curr = model(
            states=current_states, 
            offset=offset, 
            x_embed=x_out_curr, 
            return_latent=True
        )
        logit = logits[:, -1, :] / temp
        # FIX: Support batched inference - torch.argmax returns per-sample token
        # Always returns list of token ids (one per batch element)
        char_idx = torch.argmax(logit, dim=-1).cpu().tolist()
        thought_tokens.append(char_idx)
        offset += 1
        
    return current_states, thought_tokens

def think_to_english(model, states, max_bytes=200):
    # Runs latent_think to project latent thought tokens and decode them to ASCII
    _, thought_tokens = latent_think(model, states, n_thoughts=max_bytes)
    
    if not thought_tokens:
        return ""
    
    # Handle both scalar tokens (B=1) and batched (B>1)
    batch_size = len(thought_tokens[0]) if isinstance(thought_tokens[0], list) else 1
    
    chars_per_batch = [[] for _ in range(batch_size)]
    for step_tokens in thought_tokens:
        if isinstance(step_tokens, list):
            for b, token in enumerate(step_tokens):
                if 32 <= token <= 126:
                    chars_per_batch[b].append(chr(token))
                elif token == 10:
                    chars_per_batch[b].append('\n')
                elif token == 13:
                    chars_per_batch[b].append('\r')
                else:
                    chars_per_batch[b].append('.')
        else:
            # Scalar fallback (shouldn't happen with new batched latent_think)
            token = step_tokens
            if 32 <= token <= 126:
                chars_per_batch[0].append(chr(token))
            elif token == 10:
                chars_per_batch[0].append('\n')
            elif token == 13:
                chars_per_batch[0].append('\r')
            else:
                chars_per_batch[0].append('.')
    
    decoded = ["".join(chars) for chars in chars_per_batch]
    return decoded[0] if batch_size == 1 else decoded
