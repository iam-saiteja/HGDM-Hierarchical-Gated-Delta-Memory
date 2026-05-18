# HGDM Experimental Suite

This folder contains the 11 absolute core experiments designed to mathematically and empirically prove the superiority of the HGDM architecture over standard Transformers on an RTX 3090 Ti.

Each experiment is organized into its own folder for clean result management.

## Directory Structure

### `exp1_enwik8/`
**Goal**: The headline result. Proves HGDM learns faster and uses less memory than a comparable Transformer.
*   `run_exp.py`: Trains both 120M models for 1000 steps on Enwik8.
*   Outputs: `results.json`, `hgdm_enwik8_120M.pt`, `transformer_enwik8_120M.pt`.

### `exp2_memory/`
**Goal**: The $O(1)$ Memory Proof.
*   `run_exp.py`: Tests sequences from `512` up to `16,384`.
*   Forces the Transformer to use raw math (`FlashAttention` disabled) to expose the $O(N^2)$ quadratic explosion.
*   Outputs: `results.json`.

### `exp3_throughput/`
**Goal**: The Fused Kernel Speed Proof.
*   `run_exp.py`: Tests tokens processed per second across sequence lengths.
*   Outputs: `results.json`.

### `exp4_ablation/`
**Goal**: The Gating Mechanism Proof.
*   `run_exp.py`: Trains 3 variants of HGDM (Full Multi-Scale, Flat $\tau=200$, and Learned/No-Bias).
*   Outputs: `results.json`.

### `exp5_inference/`
**Goal**: The Infinite Context Qualitative Test.
*   `run_exp.py`: Loads the trained Enwik8 checkpoint and generates a 2000-byte sequence.
*   Outputs: `results.json`.

### `exp6_math/`
**Goal**: Domain Transfer / Byte-level Universality Proof.
*   `run_exp.py`: Fine-tunes the Enwik8 checkpoint on synthetic math/algebra.
*   Outputs: `results.json`.

### `exp7_multimodal/`
**Goal**: The Ultimate Multimodal / Universal Sequence Engine Proof.
*   `run_exp.py`: Trains from scratch on raw PCM (Audio), Raw RGB (Image), and Raw Frames (Video).
*   `plot_results.py`: Renders the raw byte hallucinations into a PNG figure.
*   Outputs: `results.json`, `generated_audio.raw`, `generated_image.raw`, `generated_video.raw`, `hallucination_proof.png`.

### `exp8_kernel_impact/`
**Goal**: The Nitro Engine Speedup Proof.
*   `run_exp.py`: Benchmarks the custom Triton Fused Kernel vs. a standard sequential PyTorch implementation.
*   Outputs: `results.json`.

### `exp9_long_gating/`
**Goal**: Long-Range Dependency Isolation Proof.
*   `run_exp.py`: Trains Full vs. Flat variants at 4096 context length to prove multi-scale advantage over distance.
*   Outputs: `results.json`.

### `exp10_state_stability/`
**Goal**: The Mathematical Recurrent Stability Proof.
*   `run_exp.py`: Auto-regressively generates 100,000 tokens while measuring the Frobenius norm of the state to prove it does not explode/vanish.
*   Outputs: `results.json`.

### `exp11_passkey_retrieval/`
**Goal**: The Effective Context Window Proof (Needle in a Haystack).
*   `run_exp.py`: Runs a curriculum training and evaluation loop on the "Passkey Retrieval" task up to 16,384 tokens to prove exact retrieval across extreme depths.
*   Outputs: `results.json`.

---

## How to Run
Navigate to any experiment folder and run the script:
```bash
cd exp1_enwik8
python run_exp.py
```
Each folder will contain its own clean `results.json` after execution.
