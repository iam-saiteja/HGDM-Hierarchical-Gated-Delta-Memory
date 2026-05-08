import json
import matplotlib.pyplot as plt
import numpy as np

def plot_results():
    with open('results.json', 'r') as f:
        data = json.load(f)
        
    seq_lengths = data['seq_lengths']
    hgdm_mem = [x if x is not None else np.nan for x in data['HGDM']['memory']]
    tf_mem = [x if x is not None else np.nan for x in data['Transformer']['memory']]
    
    hgdm_tp = [x if x is not None else np.nan for x in data['HGDM']['throughput']]
    tf_tp = [x if x is not None else np.nan for x in data['Transformer']['throughput']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Memory
    ax1.plot(seq_lengths, tf_mem, marker='o', label='Transformer Baseline', color='red', linestyle='--')
    ax1.plot(seq_lengths, hgdm_mem, marker='s', label='HGDM (Ours)', color='blue', linewidth=2)
    ax1.set_xlabel('Sequence Length (Tokens)')
    ax1.set_ylabel('Peak VRAM (MB)')
    ax1.set_title('Exp 1: Training Memory vs Sequence Length')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # Throughput
    ax2.plot(seq_lengths, tf_tp, marker='o', label='Transformer Baseline', color='red', linestyle='--')
    ax2.plot(seq_lengths, hgdm_tp, marker='s', label='HGDM (Ours)', color='blue', linewidth=2)
    ax2.set_xlabel('Sequence Length (Tokens)')
    ax2.set_ylabel('Throughput (Tokens/sec)')
    ax2.set_title('Exp 2: Training Throughput vs Sequence Length')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('exp1_2_results.png', dpi=300)
    print("Saved exp1_2_results.png")

if __name__ == "__main__":
    plot_results()
