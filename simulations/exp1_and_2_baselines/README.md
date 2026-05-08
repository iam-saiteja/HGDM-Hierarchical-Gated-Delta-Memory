# Experiment 1 & 2: Memory Scaling & Throughput

## Objective
This experiment empirically proves the O(1) constant memory scaling of the HGDM architecture and benchmarks its raw throughput (Tokens/sec) against a standard Transformer of identical size.

## Setup
You will run this script on your GPU server. The script generates synthetic data, so no dataset downloading is required. It tests sequence lengths from 512 up to 8192.

## How to Run on Server
1. Copy the entire `HTSPC-H3` directory to your server (or pull from your repo).
2. Navigate to this directory:
   ```bash
   cd htspc/simulations/exp1_and_2_baselines
   ```
3. Run the benchmark:
   ```bash
   python run_exp.py
   ```
4. The script will output a file named `results.json`.

## Next Steps
Once the script completes on your server, copy the `results.json` file back to this folder on your local machine. Let me know when it's here, and I will write the plotting script to analyze the real-world hardware data.
