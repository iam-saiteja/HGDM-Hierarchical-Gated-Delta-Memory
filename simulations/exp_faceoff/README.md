# Enwik8 Faceoff: HGDM vs Transformer (Raw Math)

## Objective
This experiment strips away all modern software optimizations (like FlashAttention) to expose the raw mathematical scaling of both architectures. It will prove that HGDM scales linearly ($O(N)$) during training, while the standard Transformer suffers from a catastrophic quadratic explosion ($O(N^2)$) and ultimately OOMs on consumer hardware.

## Setup
You will run this on your GPU server. The script uses synthetic data for memory/throughput testing to ensure the bottlenecks are strictly architectural, not I/O related.

## How to Run on Server
1. Navigate to this directory on your server:
   ```bash
   cd htspc/simulations/exp_faceoff
   ```
2. Run the Faceoff benchmark:
   ```bash
   python run_faceoff.py
   ```
3. The script will output a file named `faceoff_results.json`.

## Expectations
The Transformer will likely OOM at sequence length 4096 or 8192. HGDM will survive and accelerate.

## Next Steps
Once the script completes on your server, copy the `faceoff_results.json` file back to this folder on your local machine so we can plot the definitive proof!
