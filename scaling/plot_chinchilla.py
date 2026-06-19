import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_scaling():
    data_path = os.path.join(os.path.dirname(__file__), "chinchilla_scaling_data.json")
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Please run train_chinchilla.py first.")
        return
        
    with open(data_path, 'r') as f:
        data = json.load(f)
        
    params = []
    losses = []
    names = []
    
    for name, stats in data.items():
        params.append(stats['parameters'])
        losses.append(stats['val_loss'])
        names.append(name)
        
    log_params = np.log10(params)
    log_losses = np.log10(losses)
    
    z = np.polyfit(log_params, log_losses, 1)
    p = np.poly1d(z)
    
    x_line = np.linspace(min(log_params)-0.2, max(log_params)+0.2, 100)
    y_line = p(x_line)
    
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 7))
    
    plt.plot(10**x_line, 10**y_line, color='#27ae60', linestyle='--', linewidth=2.5, label=f'True Compute-Optimal Fit ($L = a P^{{{z[0]:.3f}}}$)')
    
    plt.scatter(params, losses, color='#8e44ad', s=200, zorder=5, edgecolor='black', linewidth=1.5)
    
    for i, name in enumerate(names):
        plt.annotate(f"{name}\nLoss: {losses[i]:.4f}", 
                     (params[i], losses[i]), 
                     xytext=(15, 5), textcoords='offset points', 
                     fontsize=12, fontweight='bold', color='#2c3e50')
        
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('Parameter Count (Log Scale)', fontsize=14, fontweight='bold', labelpad=15)
    plt.ylabel('Validation Cross-Entropy Loss (Log Scale)', fontsize=14, fontweight='bold', labelpad=15)
    plt.title('OmegaGDM True Chinchilla Scaling Law (20x Tokens)', fontsize=18, fontweight='bold', pad=20)
    
    plt.grid(True, which="major", ls="-", alpha=0.6)
    plt.grid(True, which="minor", ls=":", alpha=0.4)
    
    plt.legend(fontsize=14, loc='upper right', frameon=True, shadow=True)
    
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "chinchilla_scaling_law.png")
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    print(f"Professional Chinchilla scaling plot saved successfully to {out_path}")

if __name__ == "__main__":
    plot_scaling()
