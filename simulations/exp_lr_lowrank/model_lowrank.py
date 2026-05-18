import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from hgdm_ultimate import HGDMConfig

class LowRankMultiHeadGatedDelta(nn.Module):
    """Prototype low-rank HGDM head: projects into r-d latent, maintains S_r (r x r) per head.
    Sequential implementation only (no Triton fused kernel).
    """
    def __init__(self, config: HGDMConfig, r: int = 8, force_sequential=True):
        super().__init__()
        self.config = config
        self.H = config.n_heads
        self.d_k = config.d_k
        self.d_v = config.d_v
        self.r = r
        self.force_sequential = True

        # Projections to latent r-space per head
        self.W_q = nn.Linear(config.d_model, self.H * self.d_k, bias=False)
        self.W_k = nn.Linear(config.d_model, self.H * self.d_k, bias=False)
        self.W_v = nn.Linear(config.d_model, self.H * self.d_v, bias=False)

        self.Pq = nn.Linear(self.d_k, self.r, bias=False)
        self.Pk = nn.Linear(self.d_k, self.r, bias=False)
        self.Pv = nn.Linear(self.d_v, self.r, bias=False)

        # Back projection from latent r to output d_v per head
        self.W_back = nn.Linear(self.H * self.r, self.H * self.d_v, bias=False)

        self.W_alpha = nn.Linear(config.d_model, self.H)
        self.W_beta = nn.Linear(config.d_model, self.H)
        self.W_out_gate = nn.Linear(config.d_model, self.H * self.d_v)
        self.W_o = nn.Linear(self.H * self.d_v, config.d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        # mimic some of HGDM init behaviour
        with torch.no_grad():
            self.W_alpha.bias.zero_()
            self.W_alpha.weight.zero_()
            self.W_beta.weight.zero_()
            self.W_beta.bias.fill_(-1.0)
            self.W_out_gate.weight.zero_()
            self.W_out_gate.bias.zero_()

    def forward(self, x, state=None):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.H, self.d_k)
        k = self.W_k(x).view(B, T, self.H, self.d_k)
        v = self.W_v(x).view(B, T, self.H, self.d_v)

        alpha = torch.sigmoid(self.W_alpha(x))  # (B,T,H)
        beta = torch.sigmoid(self.W_beta(x))    # (B,T,H)
        out_gate = torch.sigmoid(self.W_out_gate(x)).view(B, T, self.H, self.d_v)

        # Prepare projected latent vectors
        q_r = self.Pq(q)  # not directly broadcastable: Pq expects (..., d_k)->(..., r)
        # Need to apply Pq per head: reshape to (B*T*H, d_k)
        q_flat = q.contiguous().view(-1, self.d_k)
        k_flat = k.contiguous().view(-1, self.d_k)
        v_flat = v.contiguous().view(-1, self.d_v)

        q_r_flat = self.Pq(q_flat).view(B, T, self.H, self.r)
        k_r_flat = self.Pk(k_flat).view(B, T, self.H, self.r)
        v_r_flat = self.Pv(v_flat).view(B, T, self.H, self.r)

        # Initialize latent states: S_r per head (B,H,r,r)
        if state is None:
            S = torch.zeros(B, self.H, self.r, self.r, device=x.device, dtype=x.dtype)
        else:
            S = state

        outputs = []
        for t in range(T):
            a = k_r_flat[:, t]  # (B,H,r)
            b = v_r_flat[:, t]  # (B,H,r)
            # outer product in latent space
            outer = a.unsqueeze(-1) * b.unsqueeze(-2)  # (B,H,r,r)
            S = alpha[:, t, :, None, None] * S + beta[:, t, :, None, None] * outer

            # compute output: q_r @ S -> (B,H,r) then project back to d_v
            qr = q_r_flat[:, t]  # (B,H,r)
            out_r = torch.einsum('bhr,bhrk->bhk', qr, S)  # (B,H,r)
            out_r_flat = out_r.reshape(B, 1, self.H * self.r)
            out_proj = self.W_back(out_r_flat).view(B, self.H, self.d_v)  # (B,H,d_v)
            out_proj = out_proj * out_gate[:, t]  # (B,H,d_v)
            outputs.append(out_proj)

        out = torch.stack(outputs, dim=1).reshape(B, T, -1)
        return self.W_o(out), S

class HGDMLowRankLayer(nn.Module):
    def __init__(self, config: HGDMConfig, layer_idx: int, r: int = 8):
        super().__init__()
        self.norm1 = torch.nn.Identity()
        self.mixer = LowRankMultiHeadGatedDelta(config, r=r)
        self.norm2 = torch.nn.Identity()
        self.ffn = torch.nn.Identity()

    def forward(self, x, state=None):
        m_out, new_state = self.mixer(x, state=state)
        x = x + m_out
        return x, new_state

class HGDMLowRankUltimate(nn.Module):
    def __init__(self, config: HGDMConfig, n_layers: int = 6, r: int = 8):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 4096, config.d_model) * 0.02)
        self.layers = nn.ModuleList([HGDMLowRankLayer(config, i, r=r) for i in range(n_layers)])
        self.norm_f = nn.Identity()
        self.fc_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.fc_out.weight = self.embedding.weight

    def forward(self, byte_seq, states=None):
        B, T = byte_seq.shape
        x = self.embedding(byte_seq)
        x = x + self.pos_embedding[:, :T, :]
        if states is None:
            states = [None] * len(self.layers)
        next_states = []
        for i, layer in enumerate(self.layers):
            x, ns = layer(x, states[i])
            next_states.append(ns)
        x = self.norm_f(x)
        return self.fc_out(x), next_states
