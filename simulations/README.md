# HGDM Experimental Suite

This folder contains the 5 absolute core experiments designed to mathematically and empirically prove the superiority of the HGDM architecture over standard Transformers on an RTX 3090 Ti.

Each script imports the pristine `HGDMUltimate` architecture from the root directory and runs a specific benchmark.

## The Experiments

### 1. `exp1_enwik8_main.py`
**Goal**: The headline result. Proves HGDM learns faster and uses less memory than a comparable Transformer.
*   Trains both 120M models for 1000 steps on Enwik8.
*   Logs BPB, Peak VRAM, and Training Time.

### 2. `exp2_memory_scaling.py`
**Goal**: The $O(1)$ Memory Proof.
*   Tests sequences from `512` up to `16,384`.
*   Forces the Transformer to use raw math (`FlashAttention` disabled) to expose the $O(N^2)$ quadratic explosion and trigger an OOM.
*   Proves HGDM's memory stays linear/flat.

### 3. `exp3_throughput.py`
**Goal**: The Fused Kernel Speed Proof.
*   Tests tokens processed per second across sequence lengths.
*   Shows HGDM maintaining high throughput even at large contexts.

### 4. `exp4_ablation.py`
**Goal**: The Gating Mechanism Proof.
*   Trains 3 variants of HGDM (Full Multi-Scale, Flat $\tau=200$, and Learned/No-Bias).
*   Proves that hierarchical timescales are mathematically required for optimal learning on Enwik8.

### 5. `exp5_inference.py`
**Goal**: The Infinite Context Qualitative Test.
*   Forces the model to generate a continuous 2000-byte sequence to prove the architecture handles long-range generation without OOMing.

### 6. `exp6_math_transfer.py`
**Goal**: Domain Transfer / Byte-level Universality Proof.
*   Generates synthetic math/algebra equations.
*   Loads the trained Enwik8 checkpoint, computes zero-shot BPB, and fine-tunes for 500 steps.
*   Proves the architecture adapts to radically different domains with minimal compute.

---

## How to Run
Simply execute any script from this directory:
```bash
python exp1_enwik8_main.py
```
Each script will output a `.json` file containing the precise data needed for the paper's plots and tables.
