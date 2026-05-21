import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Add current directory to path to resolve imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hgdm_omega import OmegaGDM, OmegaConfig

def auto_detect_config(checkpoint_path):
    print(f"[System] Loading checkpoint to auto-detect model config: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    d_byte = 256
    catcher_layers = 0
    renderer_layers = 0
    core_layers = 0
    n_heads = 16
    d_k = 64
    d_v = 64
    d_ff = 4096
    d_model = 1024
    decimation_rate = 8
    
    # 1. Detect catcher layers
    while f"byte_catcher.{catcher_layers}.mixer.W_q.weight" in state_dict:
        catcher_layers += 1
        
    # 2. Detect renderer layers
    while f"byte_renderer.{renderer_layers}.mixer.W_q.weight" in state_dict:
        renderer_layers += 1
        
    # 3. Detect core layers
    while f"semantic_core.{core_layers}.mixer.W_q.weight" in state_dict:
        core_layers += 1
        
    # 4. Detect d_model from decimator_proj.weight shape [d_model, d_byte]
    if "decimator_proj.weight" in state_dict:
        d_model = state_dict["decimator_proj.weight"].shape[0]
        d_byte = state_dict["decimator_proj.weight"].shape[1]
        
    # 5. Detect n_heads and d_k from semantic_core.0.mixer.W_q.weight shape [n_heads * d_k, d_model]
    if "semantic_core.0.mixer.W_q.weight" in state_dict:
        shape = state_dict["semantic_core.0.mixer.W_q.weight"].shape
        total_q_dim = shape[0]
        if total_q_dim % 64 == 0:
            d_k = 64
            n_heads = total_q_dim // 64
        else:
            d_k = 32
            n_heads = total_q_dim // 32
            
    # 6. Detect d_ff from semantic_core.0.ffn.w1.weight shape [d_ff, d_model]
    if "semantic_core.0.ffn.w1.weight" in state_dict:
        d_ff = state_dict["semantic_core.0.ffn.w1.weight"].shape[0]
        
    # 7. Check if variable delta_t is used (if W_delta is in state_dict)
    use_variable_delta_t = "semantic_core.0.mixer.W_delta.weight" in state_dict
    
    # 8. Check max_position_embeddings from semantic_pos_embed shape [1, max_pos // W, d_model]
    max_position_embeddings = 2048
    if "semantic_pos_embed" in state_dict:
        sem_pos_len = state_dict["semantic_pos_embed"].shape[1]
        max_position_embeddings = sem_pos_len * decimation_rate
        
    config = OmegaConfig(
        d_byte=d_byte,
        catcher_layers=catcher_layers,
        renderer_layers=renderer_layers,
        d_model=d_model,
        core_layers=core_layers,
        n_heads=n_heads,
        d_k=d_k,
        d_v=d_k,
        d_ff=d_ff,
        decimation_rate=decimation_rate,
        max_position_embeddings=max_position_embeddings,
        vocab_size=256,
        use_state_fusion=False,
        use_variable_delta_t=use_variable_delta_t
    )
    print(f"[System] Auto-detected Config parameters:")
    print(f"  - d_byte: {d_byte}")
    print(f"  - catcher_layers: {catcher_layers}")
    print(f"  - core_layers: {core_layers}")
    print(f"  - renderer_layers: {renderer_layers}")
    print(f"  - d_model: {d_model}")
    print(f"  - n_heads: {n_heads}")
    print(f"  - d_ff: {d_ff}")
    print(f"  - use_variable_delta_t: {use_variable_delta_t}")
    print(f"  - max_position_embeddings: {max_position_embeddings}")
    return config

def load_model(device, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint path not found: {checkpoint_path}")
        sys.exit(1)
        
    config = auto_detect_config(checkpoint_path)
    model = OmegaGDM(config).to(device)
    
    print(f"[System] Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("[System] Model state dict successfully loaded!")
    model.eval()
    return model, config

def run_interpretability_analysis(model, config, prompt_text, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    prompt_bytes = list(prompt_text.encode('utf-8', errors='ignore'))
    
    # 1. Truncate prompt if sequence length is not a multiple of decimation rate
    W = config.decimation_rate
    T = len(prompt_bytes)
    if T % W != 0:
        new_T = (T // W) * W
        if new_T == 0:
            new_T = W
            prompt_text = (prompt_text + " " * W)[:W]
            prompt_bytes = list(prompt_text.encode('utf-8', errors='ignore'))
        else:
            prompt_bytes = prompt_bytes[:new_T]
            prompt_text = bytes(prompt_bytes).decode('utf-8', errors='ignore')
        T = len(prompt_bytes)
        print(f"[Notice] Adjusted prompt length to multiple of {W} (T={T}) for perfect alignment.")
        
    x = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
    
    # Hooks storage
    alphas = {}
    betas = {}
    core_hidden_states = {}
    
    def get_mixer_hook(name, alphas_storage, betas_storage):
        def hook(module, input, output):
            x_in = input[0]
            with torch.no_grad():
                if getattr(module.config, "use_variable_delta_t", False):
                    delta_t = F.softplus(module.W_delta(x_in)) + 1e-3
                    lambdas = torch.exp(module.W_lambda)
                    alpha = torch.exp(-delta_t * lambdas[None, None, :])
                else:
                    alpha = torch.sigmoid(module.W_alpha(x_in))
                beta = torch.sigmoid(module.W_beta(x_in))
            alphas_storage[name] = alpha.detach().cpu()
            betas_storage[name] = beta.detach().cpu()
        return hook
        
    def get_hidden_hook(name, storage):
        def hook(module, input, output):
            x_semantic, _ = output
            storage[name] = x_semantic.detach().cpu()
        return hook

    # Register hooks
    registered_hooks = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "MultiHeadGatedDelta":
            h = module.register_forward_hook(get_mixer_hook(name, alphas, betas))
            registered_hooks.append(h)
            
    last_core_layer_idx = config.core_layers - 1
    last_core_layer = model.semantic_core[last_core_layer_idx]
    h = last_core_layer.register_forward_hook(get_hidden_hook("last_core", core_hidden_states))
    registered_hooks.append(h)
    
    # Run forward pass
    print(f"[System] Executing forward pass on prompt ({T} bytes)...")
    with torch.no_grad():
        logits, _ = model(x)
        
    # Remove hooks
    for h in registered_hooks:
        h.remove()
    print("[System] Forward pass completed. Analyzing activations...")
    
    # 2. Extract and average gate values per step
    # Sequence mapping
    alpha_layers = []
    beta_layers = []
    layer_names = []
    
    # Catcher
    for i in range(config.catcher_layers):
        name = f"byte_catcher.{i}.mixer"
        if name in alphas:
            a = alphas[name].mean(dim=-1).squeeze(0).numpy() # [T]
            b = betas[name].mean(dim=-1).squeeze(0).numpy() # [T]
            alpha_layers.append(a)
            beta_layers.append(b)
            layer_names.append(f"Catcher {i}")
            
    # Core (decimated by W)
    for i in range(config.core_layers):
        name = f"semantic_core.{i}.mixer"
        if name in alphas:
            # shape is [1, N, H], average to [N]
            a_sem = alphas[name].mean(dim=-1).squeeze(0).numpy()
            b_sem = betas[name].mean(dim=-1).squeeze(0).numpy()
            # repeat-interleave to match full sequence length T
            a_rep = np.repeat(a_sem, W)
            b_rep = np.repeat(b_sem, W)
            alpha_layers.append(a_rep)
            beta_layers.append(b_rep)
            layer_names.append(f"Core {i}")
            
    # Renderer
    for i in range(config.renderer_layers):
        name = f"byte_renderer.{i}.mixer"
        if name in alphas:
            a = alphas[name].mean(dim=-1).squeeze(0).numpy() # [T]
            b = betas[name].mean(dim=-1).squeeze(0).numpy() # [T]
            alpha_layers.append(a)
            beta_layers.append(b)
            layer_names.append(f"Renderer {i}")
            
    alpha_grid = np.stack(alpha_layers, axis=0) # [NumLayers, T]
    beta_grid = np.stack(beta_layers, axis=0)   # [NumLayers, T]
    
    # 3. Terminal colored output
    # Let's show Catcher average vs Core average vs Renderer average
    catcher_indices = list(range(config.catcher_layers))
    core_indices = list(range(config.catcher_layers, config.catcher_layers + config.core_layers))
    renderer_indices = list(range(config.catcher_layers + config.core_layers, len(layer_names)))
    
    avg_catcher_beta = beta_grid[catcher_indices].mean(axis=0) if catcher_indices else np.zeros(T)
    avg_core_beta = beta_grid[core_indices].mean(axis=0) if core_indices else np.zeros(T)
    avg_renderer_beta = beta_grid[renderer_indices].mean(axis=0) if renderer_indices else np.zeros(T)
    
    avg_catcher_alpha = alpha_grid[catcher_indices].mean(axis=0) if catcher_indices else np.zeros(T)
    avg_core_alpha = alpha_grid[core_indices].mean(axis=0) if core_indices else np.zeros(T)
    avg_renderer_alpha = alpha_grid[renderer_indices].mean(axis=0) if renderer_indices else np.zeros(T)
    
    def print_colored_timeline(text, values, title, cmap_name="write"):
        print(f"\n[Interpret] {title}:")
        sys.stdout.write("  ")
        for char, val in zip(text, values):
            # Safe character display
            c = char if char not in ['\n', '\r', '\t'] else ' '
            
            # Map values to ANSI colors
            if cmap_name == "write":
                # Beta gate: Write intensity. High is yellow/red, low is grey
                if val > 0.6:
                    color = "\033[91;1m" # Bold Red
                elif val > 0.4:
                    color = "\033[93m"   # Yellow
                elif val > 0.2:
                    color = "\033[92m"   # Green
                else:
                    color = "\033[90m"   # Dark Grey
            else:
                # Alpha gate: Retention. High is cyan/blue, low is white
                if val > 0.95:
                    color = "\033[96;1m" # Bold Cyan (High memory retention)
                elif val > 0.8:
                    color = "\033[94m"   # Blue
                elif val > 0.5:
                    color = "\033[97m"   # White
                else:
                    color = "\033[90m"   # Dark Grey
            
            sys.stdout.write(f"{color}{c}\033[0m")
        sys.stdout.write("\n")
        sys.stdout.flush()

    print("\n" + "="*80)
    print("                 OMEGAGDM INTERPRETABILITY TERMINAL DASHBOARD")
    print("="*80)
    print_colored_timeline(prompt_text, avg_catcher_beta, "CATCHER WRITE INTENSITY (Beta)")
    print_colored_timeline(prompt_text, avg_core_beta, "SEMANTIC CORE WRITE INTENSITY (Beta)")
    print_colored_timeline(prompt_text, avg_renderer_beta, "RENDERER WRITE INTENSITY (Beta)")
    print("-" * 80)
    print_colored_timeline(prompt_text, avg_catcher_alpha, "CATCHER STATE RETENTION (Alpha)", cmap_name="read")
    print_colored_timeline(prompt_text, avg_core_alpha, "SEMANTIC CORE STATE RETENTION (Alpha)", cmap_name="read")
    print_colored_timeline(prompt_text, avg_renderer_alpha, "RENDERER STATE RETENTION (Alpha)", cmap_name="read")
    print("="*80)

    # 4. Generate Heatmap Figure using Matplotlib
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Retention Heatmap
    im1 = axes[0].imshow(alpha_grid, aspect='auto', cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    axes[0].set_title("Retention Gating Signal (Alpha) across Manifolds", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Layer Manifold", fontsize=12)
    axes[0].set_yticks(range(len(layer_names)))
    axes[0].set_yticklabels(layer_names, fontsize=8)
    fig.colorbar(im1, ax=axes[0], label="Alpha Value")
    
    # Write Heatmap
    im2 = axes[1].imshow(beta_grid, aspect='auto', cmap='magma', interpolation='nearest', vmin=0, vmax=1)
    axes[1].set_title("Write Gating Signal (Beta) across Manifolds", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Layer Manifold", fontsize=12)
    axes[1].set_yticks(range(len(layer_names)))
    axes[1].set_yticklabels(layer_names, fontsize=8)
    axes[1].set_xlabel("Token Sequence / Character", fontsize=12)
    fig.colorbar(im2, ax=axes[1], label="Beta Value")
    
    # Display prompt string as X-ticks
    plt.xticks(range(T), list(prompt_text), rotation=90, fontfamily='monospace', fontsize=9)
    plt.tight_layout()
    heatmap_path = os.path.join(out_dir, "gate_heatmaps.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"[Interpret] Gating heatmaps saved to: {heatmap_path}")
    
    # 5. PCA Trajectory on Semantic Hidden States
    if "last_core" in core_hidden_states:
        X_sem = core_hidden_states["last_core"].squeeze(0) # [N, d_model]
        N = X_sem.shape[0]
        
        # Center the semantic states
        X_mean = X_sem.mean(dim=0, keepdim=True)
        X_centered = X_sem - X_mean
        
        # Compute low-rank PCA projection using PyTorch's native SVD
        U, S, V = torch.pca_lowrank(X_centered, q=2)
        coords = torch.matmul(X_centered, V[:, :2]).numpy() # [N, 2]
        
        plt.figure(figsize=(10, 8))
        plt.plot(coords[:, 0], coords[:, 1], 'o--', color='#2ca02c', alpha=0.6, markersize=8, label='Semantic State Path')
        
        # Draw path direction arrows
        for i in range(N - 1):
            plt.annotate('', xy=coords[i+1], xytext=coords[i],
                         arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5, ls='-', shrinkA=4, shrinkB=4))
                         
        # Label each coordinate with the corresponding 8-byte text window
        for i in range(N):
            chunk_bytes = prompt_bytes[i*W : (i+1)*W]
            chunk_repr = bytes(chunk_bytes).decode('utf-8', errors='replace')
            # clean representation for label
            chunk_repr = chunk_repr.replace('\n', '\\n').replace('\t', '\\t')
            plt.annotate(f"[{i}] '{chunk_repr}'", xy=coords[i], xytext=(6, 6), textcoords='offset points',
                         fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.3))
                         
        plt.title("Semantic Core State Trajectory (2D PCA projection)", fontsize=14, fontweight='bold')
        plt.xlabel("Principal Component 1", fontsize=12)
        plt.ylabel("Principal Component 2", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        
        trajectory_path = os.path.join(out_dir, "semantic_trajectory.png")
        plt.savefig(trajectory_path, dpi=150)
        plt.close()
        print(f"[Interpret] Semantic state trajectory PCA plot saved to: {trajectory_path}")
    else:
        print("[ERROR] Semantic core state was not captured by hook.")
        
    # 6. Bilinear Highway Parameter Extraction
    print("\n" + "="*80)
    print("                 FACTORED BILINEAR STATE HIGHWAYS DIAGNOSTIC")
    print("="*80)
    if hasattr(model, "highway_td_gate") and hasattr(model, "highway_bu_gate"):
        # Top-down gates (Core -> Renderer)
        td_val = torch.sigmoid(model.highway_td_gate).detach().cpu().numpy()
        print(f"Top-Down Highway (Core -> Renderer, {len(td_val)} heads):")
        for h, val in enumerate(td_val):
            print(f"  Head {h:02d}: {val:.4f} " + ("█" * int(val * 20)))
            
        print("-" * 80)
        # Bottom-up gates (Renderer -> Core)
        bu_val = torch.sigmoid(model.highway_bu_gate).detach().cpu().numpy()
        print(f"Bottom-Up Highway (Renderer -> Core, {len(bu_val)} heads):")
        for h, val in enumerate(bu_val):
            print(f"  Head {h:02d}: {val:.4f} " + ("█" * int(val * 20)))
    else:
        print("[Interpret] Factored Bilinear State Highways parameters not found on model.")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="OmegaGDM Interpretability Tool")
    parser.add_argument("--checkpoint", type=str, default="hgdm_1b_latest.pt", help="Path to checkpoint file")
    parser.add_argument("--prompt", type=str, default="The theory of relativity states that space and time are relative. def fibonacci(n):", help="Prompt text to analyze")
    parser.add_argument("--out-dir", type=str, default="interpretability_results", help="Output folder to save figures")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Running on device: {device}")
    
    model, config = load_model(device, args.checkpoint)
    run_interpretability_analysis(model, config, args.prompt, device, args.out_dir)

if __name__ == "__main__":
    main()
