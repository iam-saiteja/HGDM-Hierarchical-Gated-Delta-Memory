import json
import matplotlib.pyplot as plt
import numpy as np

def plot_stress():
    with open('stress_results.json', 'r') as f:
        data = json.load(f)
        
    seq_lengths = [str(x) for x in data['seq_lengths']]
    memory = data['memory']
    status = data['status']
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['green' if s == 'SUCCESS' else 'red' for s in status]
    bars = ax.bar(seq_lengths, [m if m is not None else 24000 for m in memory], color=colors, alpha=0.7)
    
    # Add text labels
    for bar, mem, stat in zip(bars, memory, status):
        height = bar.get_height()
        if stat == 'SUCCESS':
            ax.text(bar.get_x() + bar.get_width()/2., height - 1000,
                    f'{mem:.0f} MB', ha='center', va='bottom', color='white', fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height / 2,
                    'OOM\n(>24GB)', ha='center', va='center', color='white', fontweight='bold', fontsize=14)
            
    ax.axhline(y=24000, color='red', linestyle='--', label='24GB GPU Limit')
    
    ax.set_xlabel('Sequence Length (Tokens)')
    ax.set_ylabel('Peak VRAM (MB)')
    ax.set_title('Exp 4: Hardware Stress Test (120M Model without Checkpointing)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('stress_plot.png', dpi=300)
    print("Saved stress_plot.png")

if __name__ == "__main__":
    plot_stress()
