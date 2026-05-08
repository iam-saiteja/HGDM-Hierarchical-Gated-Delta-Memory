import json
import matplotlib.pyplot as plt

def plot_ablation():
    with open('ablation_results.json', 'r') as f:
        data = json.load(f)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'full': 'blue', 'flat': 'orange', 'learned': 'red'}
    labels = {'full': 'HGDM (Multi-scale $\\tau$)', 'flat': 'HGDM (Flat $\\tau=200$)', 'learned': 'HGDM (No Init Bias)'}
    
    for mode in ['full', 'flat', 'learned']:
        steps = [item[0] for item in data[mode]['bpbs']]
        bpbs = [item[1] for item in data[mode]['bpbs']]
        ax.plot(steps, bpbs, label=labels[mode], color=colors[mode], linewidth=2, marker='.')
        
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Bits Per Byte (BPB)')
    ax.set_title('Exp 3: Multi-Scale Gating Ablation (TinyShakespeare)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Zoom in on the final convergence area
    plt.ylim(2.4, 4.0)
    
    plt.tight_layout()
    plt.savefig('ablation_plot.png', dpi=300)
    print("Saved ablation_plot.png")

if __name__ == "__main__":
    plot_ablation()
