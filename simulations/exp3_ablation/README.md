# Experiment 3: Multi-Scale Gating Ablation

## Objective
This experiment proves that the hierarchical $\tau$ (tau) initialization is critical for the HGDM architecture's performance. It trains three separate 120M models on a small dataset (TinyShakespeare) for 2,000 steps to compare their learning curves.

### The Three Variants:
1. **Full HGDM (Baseline)**: Multi-scale $\tau$ initialization (taus spread across heads: [4.0, 30.0, 200.0, 1200.0, 8000.0]).
2. **Flat HGDM**: All heads are forced to the same $\tau$ scale (200.0).
3. **Learned HGDM**: No initialization bias is provided (random/zero initialized gates).

## Setup
You will run this script on your GPU server. The script uses the lightweight `TinyShakespeare` dataset for rapid iteration.

## How to Run on Server
1. Navigate to this directory on your server:
   ```bash
   cd htspc/simulations/exp3_ablation
   ```
2. Run the ablation benchmark:
   ```bash
   python run_exp.py
   ```
3. The script will train three models sequentially. It will take roughly 1-2 hours on a 3090 Ti. It will output a file named `ablation_results.json` containing the loss and BPB curves.

## Next Steps
Once the script completes on your server, copy `ablation_results.json` back to this folder on your local machine. I will then generate the loss curve comparison graph for the paper.
