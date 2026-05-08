# Experiment 4: Hardware Stress Test (Long Context)

## Objective
This experiment pushes the HGDM architecture to its absolute limits on a single consumer GPU (24GB). It attempts to process extremely long sequences (16k, 32k, 65k, and 131k tokens) to demonstrate the hardware resilience and constant-memory (O(1)) properties of the fused Triton kernel compared to standard attention.

## Setup
You will run this script on your GPU server. The script uses synthetic byte data to stress-test the memory allocation during the forward and backward passes.

## How to Run on Server
1. Navigate to this directory on your server:
   ```bash
   cd htspc/simulations/exp4_stress_test
   ```
2. Run the stress test:
   ```bash
   python run_exp.py
   ```
3. The script will output `stress_results.json`.

## Expectations
A standard 120M Transformer will Out-Of-Memory (OOM) on a 24GB GPU long before reaching 32k context. The goal here is to see if the HGDM can survive 65k or even 131k context lengths.

## Next Steps
Once the script completes on your server, copy `stress_results.json` back to this folder on your local machine. I will then generate the final bar charts for the paper.
