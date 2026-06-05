"""
interpret_omega.py — HGDM / OmegaGDM Interpretability Probe
============================================================
Runs a forward pass on a given prompt, captures internal activations
(alpha/decay, beta/write-gate, out-gate, state norms, decimation events),
and writes a JSON file.

Usage
-----
  # Probe a saved OmegaGDM checkpoint:
  python interpret_omega.py --model omega --ckpt path/to/checkpoint.pt \
      --prompt "The capital of France is" --out probe_data.json

  # Probe a freshly-initialised model (demo / sanity check):
  python interpret_omega.py --model omega --prompt "Hello world" --out probe_data.json

  # Probe the vanilla TopTransformer:
  python interpret_omega.py --model transformer --prompt "Hello world" --out probe_data.json
"""

import torch
import torch.nn.functional as F
import json
import sys
import os
import math
import argparse
import numpy as np
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Model imports ─────────────────────────────────────────────────────────────
from hgdm_omega import OmegaGDM, OmegaConfig
from ultimate.hgdm_ultimate import HGDMUltimate, HGDMConfig, MultiHeadGatedDelta

# We need a TopTransformer too — import from train_omega so we don't duplicate
# the class definition.
try:
    from train_omega import TopTransformer
except ImportError:
    TopTransformer = None


# ── Activation capture ────────────────────────────────────────────────────────

class GDMProbe:
    """Attaches forward hooks to every MultiHeadGatedDelta in the model and
    records alpha, beta, out_gate, and per-timestep state norms.

    After calling model.forward() once, access .records for the data.
    """

    def __init__(self, model: torch.nn.Module):
        self.records: Dict[str, List] = {}
        self._handles = []
        self._attach(model)

    def _attach(self, model: torch.nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, MultiHeadGatedDelta):
                handle = module.register_forward_hook(self._make_hook(name))
                self._handles.append(handle)

    def _make_hook(self, name: str):
        def hook(module, inputs, outputs):
            # outputs = (out_tensor, final_state)
            # We need the intermediate activations; we re-compute them from
            # the saved input tensors (inputs[0] = x).
            x = inputs[0].detach().float()
            B, T, _ = x.shape
            H = module.H

            with torch.no_grad():
                if getattr(module.config, "use_variable_delta_t", False):
                    delta_t = F.softplus(module.W_delta(x)) + 1e-3
                    lambdas = torch.exp(module.W_lambda)
                    alpha = torch.exp(-delta_t * lambdas[None, None, :])
                else:
                    alpha = torch.sigmoid(module.W_alpha(x))  # (B, T, H)

                beta     = torch.sigmoid(module.W_beta(x))    # (B, T, H)
                out_gate = torch.sigmoid(module.W_out_gate(x)).view(B, T, H, module.d_v)

                # State norm trajectory (sequential, O(T) but fine for probing)
                q = module.W_q(x).view(B, T, H, module.d_k)
                k = module.W_k(x).view(B, T, H, module.d_k)
                v = module.W_v(x).view(B, T, H, module.d_v)

                S = torch.zeros(B, H, module.d_k, module.d_v, device=x.device)
                state_norms = []
                write_magnitudes = []
                for t in range(T):
                    delta = torch.einsum('bhk,bhd->bhkd', k[:, t], v[:, t])
                    S = alpha[:, t, :, None, None] * S + beta[:, t, :, None, None] * delta
                    # Per-head Frobenius norm
                    norm = S.reshape(B, H, -1).norm(dim=-1)      # (B, H)
                    state_norms.append(norm[0])    # take batch 0
                    # Write magnitude = beta * ||k_t|| * ||v_t||
                    wm = beta[:, t] * k[:, t].norm(dim=-1) * v[:, t].norm(dim=-1)
                    write_magnitudes.append(wm[0])
                
                state_norms = torch.stack(state_norms).cpu().tolist()
                write_magnitudes = torch.stack(write_magnitudes).cpu().tolist()

            self.records[name] = {
                "alpha":            alpha[0].cpu().tolist(),         # (T, H)
                "beta":             beta[0].cpu().tolist(),          # (T, H)
                "out_gate_norm":    out_gate[0].norm(dim=-1).cpu().tolist(),  # (T, H)
                "state_norms":      state_norms,                     # (T, H)
                "write_magnitudes": write_magnitudes,                # (T, H)
                "n_heads": H,
                "d_k": module.d_k,
                "d_v": module.d_v,
            }
        return hook

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ── OmegaGDM decimation tracker ───────────────────────────────────────────────

class OmegaDecimationProbe:
    """Hooks into OmegaGDM.forward to record which byte positions triggered
    a semantic-core update (decimation events) and the broadcast signal norms."""

    def __init__(self, model: OmegaGDM):
        self.decimation_events: List[int] = []    # byte positions of semantic updates
        self.broadcast_norms: List[float] = []    # per-byte || z_broadcast ||
        self._model = model
        self._orig_forward = model.forward
        model.forward = self._patched_forward

    def _patched_forward(self, byte_seq, states=None, offset=0):
        B, T = byte_seq.shape
        W = self._model.W

        # Run actual forward
        logits, next_states = self._orig_forward(byte_seq, states, offset)

        # Compute decimation positions
        N = T // W
        for n in range(N):
            self.decimation_events.append((n + 1) * W - 1)   # last byte of window n

        return logits, next_states

    def detach(self):
        self._model.forward = self._orig_forward


# ── Timescale analysis ────────────────────────────────────────────────────────

def compute_timescales(records: Dict) -> Dict:
    """For each layer, compute per-head average timescale τ = -1 / log(mean_α)."""
    timescales = {}
    for name, rec in records.items():
        alpha_arr = np.array(rec["alpha"])        # (T, H)
        mean_alpha = alpha_arr.mean(axis=0)       # (H,)
        # Clip to avoid log(0) or log(1)
        mean_alpha = np.clip(mean_alpha, 1e-6, 1.0 - 1e-6)
        tau = -1.0 / np.log(mean_alpha)           # (H,)
        timescales[name] = {
            "mean_alpha": mean_alpha.tolist(),
            "tau": tau.tolist(),
        }
    return timescales


# ── SVD of final state ────────────────────────────────────────────────────────

def compute_state_svd(model, byte_seq, device) -> Dict:
    """Runs a forward pass in sequential mode to get the final state S,
    then computes its per-head singular values to show information compression."""
    svd_data = {}
    # We only do this for HGDMUltimate / OmegaGDM via the sequential fallback
    if isinstance(model, (HGDMUltimate,)):
        with torch.no_grad():
            _, states = model(byte_seq)
        for i, s in enumerate(states):
            if s is None:
                continue
            # s: (B, H, d_k, d_v)
            sv_list = []
            for h in range(s.shape[1]):
                mat = s[0, h].float()  # (d_k, d_v)
                sv = torch.linalg.svdvals(mat).cpu().tolist()
                sv_list.append(sv)
            svd_data[f"layer_{i}"] = sv_list
    return svd_data


# ── Token highlight (write salience) ─────────────────────────────────────────

def compute_write_salience(records: Dict, text_bytes: List[int]) -> List[Dict]:
    """For each token position, aggregate the write magnitude across all layers and heads."""
    if not records:
        return []
    T = len(text_bytes)
    agg = np.zeros(T)
    for name, rec in records.items():
        wm = np.array(rec["write_magnitudes"])    # (T_layer, H)
        mean_wm = wm.mean(axis=-1)                # (T_layer,)
        T_layer = len(mean_wm)
        if T_layer == T:
            agg += mean_wm
        elif T_layer < T and T_layer > 0:
            # Decimated core layers have smaller sequence length (e.g., T_layer = T // W).
            # We upsample by repeating each element.
            W = T // T_layer
            upsampled = np.repeat(mean_wm, W)
            if len(upsampled) < T:
                upsampled = np.pad(upsampled, (0, T - len(upsampled)), mode='edge')
            else:
                upsampled = upsampled[:T]
            agg += upsampled
        elif T_layer > T:
            agg += mean_wm[:T]
    
    agg /= len(records)                           # average over layers
    # Normalise to [0, 1]
    mn, mx = agg.min(), agg.max()
    if mx > mn:
        agg = (agg - mn) / (mx - mn)
    result = []
    for t, byte_val in enumerate(text_bytes):
        try:
            ch = bytes([byte_val]).decode("utf-8", errors="replace")
        except Exception:
            ch = "?"
        result.append({"pos": t, "char": ch, "byte": byte_val, "salience": float(agg[t])})
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def build_omega_model(ckpt_path: Optional[str], device: torch.device) -> OmegaGDM:
    cfg = OmegaConfig(
        d_byte=256, catcher_layers=2, renderer_layers=2,
        d_model=768, core_layers=12, n_heads=12,
        d_k=64, d_v=64, d_ff=3072,
        decimation_rate=8, max_position_embeddings=2048,
        vocab_size=256, use_state_fusion=False,
    )
    
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        
        # Dynamically auto-detect model configuration sizes from loaded state dict keys
        try:
            if 'decimator_proj.weight' in sd:
                cfg.d_model = sd['decimator_proj.weight'].shape[0]
            elif 'semantic_pos_embed' in sd:
                cfg.d_model = sd['semantic_pos_embed'].shape[2]

            core_indices = [int(k.split(".")[1]) for k in sd.keys() if k.startswith("semantic_core.") and ".mixer.W_q.weight" in k]
            if core_indices:
                cfg.core_layers = max(core_indices) + 1

            catcher_indices = [int(k.split(".")[1]) for k in sd.keys() if k.startswith("byte_catcher.") and ".mixer.W_q.weight" in k]
            if catcher_indices:
                cfg.catcher_layers = max(catcher_indices) + 1

            renderer_indices = [int(k.split(".")[1]) for k in sd.keys() if k.startswith("byte_renderer.") and ".mixer.W_q.weight" in k]
            if renderer_indices:
                cfg.renderer_layers = max(renderer_indices) + 1

            first_core_prefix = "semantic_core.0"
            if f"{first_core_prefix}.mixer.W_alpha.weight" in sd:
                cfg.n_heads = sd[f"{first_core_prefix}.mixer.W_alpha.weight"].shape[0]
            if f"{first_core_prefix}.mixer.W_q.weight" in sd:
                cfg.d_k = sd[f"{first_core_prefix}.mixer.W_q.weight"].shape[0] // cfg.n_heads
            if f"{first_core_prefix}.mixer.W_v.weight" in sd:
                cfg.d_v = sd[f"{first_core_prefix}.mixer.W_v.weight"].shape[0] // cfg.n_heads
            if f"{first_core_prefix}.ffn.w1.weight" in sd:
                cfg.d_ff = sd[f"{first_core_prefix}.ffn.w1.weight"].shape[0]
            if 'semantic_pos_embed' in sd:
                cfg.max_position_embeddings = sd['semantic_pos_embed'].shape[1] * cfg.decimation_rate

            print(f"[Probe] Auto-detected model config from checkpoint:")
            print(f"        d_model={cfg.d_model}, core_layers={cfg.core_layers}, catcher_layers={cfg.catcher_layers}, renderer_layers={cfg.renderer_layers}")
            print(f"        n_heads={cfg.n_heads}, d_k={cfg.d_k}, d_v={cfg.d_v}, d_ff={cfg.d_ff}, max_pos={cfg.max_position_embeddings}")
        except Exception as e:
            print(f"[Probe] Failed to auto-detect model config from checkpoint, using defaults: {e}")

        model = OmegaGDM(cfg, force_sequential=True).to(device)
        model.load_state_dict(sd, strict=False)
        print(f"[Probe] Loaded checkpoint: {ckpt_path}")
    else:
        print("[Probe] No checkpoint — using random init (for demo / structure inspection).")
        model = OmegaGDM(cfg, force_sequential=True).to(device)
    return model


def build_transformer_model(ckpt_path: Optional[str], device: torch.device):
    if TopTransformer is None:
        raise ImportError("Could not import TopTransformer from train_omega.py")
    model = TopTransformer(vocab_size=256, d_model=768, n_layers=12, n_heads=12, max_seq_len=2048).to(device)
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        model.load_state_dict(sd, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser(description="HGDM Interpretability Probe")
    parser.add_argument("--model",    default="omega",       choices=["omega", "transformer"])
    parser.add_argument("--ckpt",     default=None,          help="Path to checkpoint .pt")
    parser.add_argument("--prompt",   default="prompt_512.txt", help="Prompt string or path to a file containing the prompt")
    parser.add_argument("--max-gen",  default=64,  type=int, help="Bytes to generate after prompt")
    parser.add_argument("--temp",     default=0.8, type=float)
    parser.add_argument("--out",      default="probe_data_512.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Probe] Device: {device}")

    # Build model
    if args.model == "omega":
        model = build_omega_model(args.ckpt, device)
        model_type = "OmegaGDM"
        decimation_rate = model.config.decimation_rate
    else:
        model = build_transformer_model(args.ckpt, device)
        model_type = "TopTransformer"
        decimation_rate = None

    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"[Probe] Model: {model_type}  |  Parameters: {params/1e6:.2f}M")

    # Load prompt (check if file path first)
    prompt_str = args.prompt
    if os.path.exists(prompt_str):
        print(f"[Probe] Loading prompt from file: {prompt_str}")
        with open(prompt_str, "rb") as f:
            prompt_bytes = list(f.read())
        prompt_str = bytes(prompt_bytes).decode("utf-8", errors="replace")
    else:
        # Fallback default 512-byte string if default file is missing and prompt_str matches default
        if prompt_str == "prompt_512.txt":
            prompt_str = (
                '<page>\n'
                '  <title>DeepMind</title>\n'
                '  <revision>\n'
                '    <text xml:space="preserve">Google DeepMind is a pioneer in the field of artificial intelligence. By combining standard neural networks with advanced reinforcement learning techniques and search algorithms, they have solved some of the most complex scientific challenges, including protein folding structure prediction and general multi-task learning.                                                                                   </text>\n'
                '  </revision>\n'
                '</page>'
            )
        prompt_bytes = list(prompt_str.encode("utf-8", errors="replace"))

    prompt_tensor = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
    print(f"[Probe] Prompt: {prompt_str!r}  ({len(prompt_bytes)} bytes)")

    # Attach probes
    gdm_probe = GDMProbe(model) if args.model == "omega" else None
    if gdm_probe is None and args.model == "omega":
        gdm_probe = GDMProbe(model)

    if args.model != "omega":
        # Still attach GDM probe; TopTransformer has no GDM layers so records will be empty
        gdm_probe = GDMProbe(model)

    dec_probe = None
    if args.model == "omega" and isinstance(model, OmegaGDM):
        dec_probe = OmegaDecimationProbe(model)

    # Forward pass (prompt only — captures activations)
    with torch.no_grad():
        logits, _ = model(prompt_tensor)

    gdm_records = dict(gdm_probe.records) if gdm_probe else {}
    gdm_probe.detach()

    # Generate text
    print(f"[Probe] Generating {args.max_gen} bytes...")
    with torch.no_grad():
        generated = model.generate(prompt_tensor, max_new_bytes=args.max_gen, temp=args.temp)
    gen_bytes  = generated[0, len(prompt_bytes):].cpu().tolist()
    gen_text   = bytes(gen_bytes).decode("utf-8", errors="replace")
    full_bytes = prompt_bytes + gen_bytes
    full_text  = bytes(full_bytes).decode("utf-8", errors="replace")
    print(f"[Probe] Generated: {gen_text!r}")

    # Timescales
    timescales = compute_timescales(gdm_records)

    # Write salience
    write_salience = compute_write_salience(gdm_records, prompt_bytes)

    # Decimation events
    dec_events = dec_probe.decimation_events if dec_probe else []
    dec_probe.detach() if dec_probe else None

    # Summary of per-layer stats
    layer_summaries = {}
    for name, rec in gdm_records.items():
        alpha_arr = np.array(rec["alpha"])
        beta_arr  = np.array(rec["beta"])
        og_arr    = np.array(rec["out_gate_norm"])
        layer_summaries[name] = {
            "mean_alpha": float(alpha_arr.mean()),
            "std_alpha":  float(alpha_arr.std()),
            "mean_beta":  float(beta_arr.mean()),
            "mean_outgate": float(og_arr.mean()),
        }

    # Assemble output
    output = {
        "meta": {
            "model_type":       model_type,
            "parameters_M":     round(params / 1e6, 3),
            "prompt":           prompt_str,
            "generated_text":   gen_text,
            "full_text":        full_text,
            "prompt_len":       len(prompt_bytes),
            "generated_len":    len(gen_bytes),
            "decimation_rate":  decimation_rate,
            "n_layers_probed":  len(gdm_records),
        },
        "layer_activations":  gdm_records,     # per-layer: alpha, beta, out_gate, state_norms
        "timescales":         timescales,       # per-layer per-head tau
        "write_salience":     write_salience,   # per-token salience score
        "decimation_events":  dec_events,       # byte positions with semantic updates
        "layer_summaries":    layer_summaries,
    }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[Probe] Saved interpretability data → {args.out}")
    print("\n============================================================\n[PROBING RESULTS SUMMARY]\n============================================================")
    for name, summary in layer_summaries.items():
        print(f"Layer: {name:<25} | Mean Alpha: {summary['mean_alpha']:.4f} (std: {summary['std_alpha']:.4f}) | Mean Beta: {summary['mean_beta']:.4f} | Mean Out-Gate: {summary['mean_outgate']:.4f}")
    
    if timescales:
        print("\nTimescales (tau = -1 / log(mean_alpha)) per layer (averaged over heads):")
        for name, ts in timescales.items():
            avg_tau = sum(ts["tau"]) / len(ts["tau"])
            print(f"Layer: {name:<25} | Average Head Tau: {avg_tau:.2f} steps (min: {min(ts['tau']):.2f}, max: {max(ts['tau']):.2f})")
    
    if write_salience:
        print("\nTop 10 highest-salience tokens in the prompt:")
        sorted_salience = sorted(write_salience, key=lambda x: x["salience"], reverse=True)
        for entry in sorted_salience[:10]:
            print(f"  Pos: {entry['pos']:2d} | Char: {entry['char']!r} | Byte: {entry['byte']:3d} | Salience: {entry['salience']:.4f}")
    print("============================================================\n")


if __name__ == "__main__":
    main()
