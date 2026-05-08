import matplotlib.pyplot as plt
import numpy as np

def plot_memory_and_speed():
    # Data from ultimate_enwik8_results.md
    seq_lengths = [100, 250, 500, 1000, 2000, 5000, 10000]
    
    tf_speed = [774, 754, 641, 482, 315, 154, 83]
    hgdm_speed = [293, 303, 305, 306, 307, 306, 307]
    
    tf_vram = [1375, 1396, 1454, 1503, 1643, 2066, 2801]
    hgdm_vram = [1365, 1365, 1365, 1365, 1365, 1365, 1365]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Memory Plot
    ax1.plot(seq_lengths, tf_vram, marker='o', label='Transformer Baseline', color='red', linestyle='--')
    ax1.plot(seq_lengths, hgdm_vram, marker='s', label='HGDM (Ours)', color='blue', linewidth=2)
    ax1.set_xlabel('Sequence Length (Bytes)')
    ax1.set_ylabel('VRAM Usage (MB)')
    ax1.set_title('Exp 1: Memory Scaling vs Sequence Length')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Speed Plot
    ax2.plot(seq_lengths, tf_speed, marker='o', label='Transformer Baseline', color='red', linestyle='--')
    ax2.plot(seq_lengths, hgdm_speed, marker='s', label='HGDM (Ours)', color='blue', linewidth=2)
    ax2.set_xlabel('Sequence Length (Bytes)')
    ax2.set_ylabel('Throughput (Bytes/sec)')
    ax2.set_title('Exp 2: Inference Throughput vs Sequence Length')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('memory_and_throughput.png', dpi=300)
    print("Saved memory_and_throughput.png")

def plot_learning_curve():
    # Data from ultimate_enwik8_results.md
    steps = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
    hgdm_bpb = [27.6456, 3.9090, 3.0238, 2.5116, 2.4234, 2.3502, 2.2260, 2.1780, 2.1493, 2.1328, 2.0930, 2.1961, 2.0253, 1.6964, 2.0841, 1.8659, 1.8629, 1.8719, 1.8095, 1.8661, 1.7388]
    tf_baseline_val_bpb = 2.8004
    hgdm_val_bpb = 1.8053

    plt.figure(figsize=(8, 5))
    plt.plot(steps, hgdm_bpb, marker='.', label='HGDM Training BPB', color='blue')
    plt.axhline(y=tf_baseline_val_bpb, color='red', linestyle='--', label=f'Transformer Final Val BPB ({tf_baseline_val_bpb})')
    plt.axhline(y=hgdm_val_bpb, color='green', linestyle='-.', label=f'HGDM Final Val BPB ({hgdm_val_bpb})')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Bits Per Byte (BPB)')
    plt.title('HGDM Training Convergence on Enwik8 (120M params)')
    plt.ylim(1.5, 4.0) # Zoom in to show the crossing point clearly
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=300)
    print("Saved learning_curve.png")

if __name__ == "__main__":
    plot_memory_and_speed()
    plot_learning_curve()
