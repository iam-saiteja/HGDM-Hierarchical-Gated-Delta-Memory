# Enwik8 Ablation: Multi-Scale Gating

## Objective
This experiment proves that the hierarchical $\tau$ (tau) initialization is critical for the HGDM architecture's performance on a massive, real-world dataset (Enwik8). 

### The Three Variants:
1. **Full HGDM (Baseline)**: Multi-scale $\tau$ initialization (taus spread across heads: [4.0, 30.0, 200.0, 1200.0, 8000.0]).
2. **Flat HGDM**: All heads are forced to the same $\tau$ scale (200.0).
3. **Learned HGDM**: No initialization bias is provided (random/zero initialized gates).

## Setup
You will run this script on your GPU server. The script automatically downloads the 100MB Enwik8 dataset.

## How to Run on Server
1. Navigate to this directory on your server:
   ```bash
   cd htspc/simulations/exp_ablation_enwik8
   ```
2. Run the ablation benchmark:
   ```bash
   python run_ablation.py
   ```
3. The script will train three 120M models sequentially for 2,000 steps each. It will output a file named `ablation_enwik8_results.json` containing the loss and BPB curves.

## Expectations
The Full HGDM should consistently achieve the lowest BPB, proving that mathematically spacing the memory decay rates is essential for learning complex linguistic structures.

## Next Steps
Once the script completes on your server, copy `ablation_enwik8_results.json` back to this folder on your local machine. I will then generate the final BPB curve for the paper.
