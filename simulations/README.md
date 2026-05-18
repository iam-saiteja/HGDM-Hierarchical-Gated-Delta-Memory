# HGDM Experimental Suite

This folder contains the core experiments designed to benchmark the empirical performance of the HGDM architecture against standard Transformers on an RTX 3090 Ti.

Each experiment is organized into its own folder for clean result management.

## Directory Structure

### `exp1_enwik8/`
**Goal**: The headline result. Demonstrates HGDM learns faster and uses less memory than a comparable Transformer.
*   `run_exp.py`: Trains both 120M models for 1000 steps on Enwik8.
*   Outputs: `results.json`, `hgdm_enwik8_120M.pt`, `transformer_enwik8_120M.pt`.

### `exp2_memory/`
**Goal**: The Memory Scaling Validation.
*   `run_exp.py`: Tests sequences from `512` up to `16,384`.
*   Benchmarks memory growth against sequence length to expose the Transformer's $O(N^2)$ quadratic explosion.
*   Outputs: `results.json`.

### `exp3_throughput/`
**Goal**: The Fused Kernel Speed Benchmark.
*   `run_exp.py`: Tests tokens processed per second across sequence lengths.
*   Outputs: `results.json`.

### `exp4_ablation/`
**Goal**: The Gating Mechanism Validation.
*   `run_exp.py`: Trains 3 variants of HGDM (Full Multi-Scale, Flat $\tau=200$, and Learned/No-Bias).
*   Outputs: `results.json`.

### `exp5_inference/`
**Goal**: The Long-Range Generative Qualitative Test.
*   `run_exp.py`: Loads the trained Enwik8 checkpoint and generates 2000 bytes.
*   Outputs: `results.json`.

### `exp6_math/`
**Goal**: Domain Transfer / Byte-level Universality Validation.
*   `run_exp.py`: Fine-tunes the Enwik8 checkpoint on synthetic math/algebra.
*   Outputs: `results.json`.

### `exp7_multimodal/`
**Goal**: The Multimodal / Universal Sequence Engine Demonstration.
*   `run_exp.py`: Trains from scratch on raw PCM (Audio), Raw RGB (Image), and Raw Frames (Video).
*   `plot_results.py`: Renders the raw byte hallucinations into a PNG figure.
*   Outputs: `results.json`, `generated_audio.raw`, `generated_image.raw`, `generated_video.raw`, `hallucination_proof.png`.

### `exp8_kernel_impact/`
**Goal**: The Nitro Engine Speedup Benchmark.
*   `run_exp.py`: Benchmarks the custom Triton Fused Kernel vs. a standard sequential PyTorch implementation.
*   Outputs: `results.json`.

### `exp9_long_gating/`
**Goal**: Long-Range Dependency Isolation Validation.
*   `run_exp.py`: Trains Full vs. Flat variants at 4096 context length to demonstrate multi-scale advantage over distance.
*   Outputs: `results.json`.

### `exp10_state_stability/`
**Goal**: The Recurrent Stability Validation.
*   `run_exp.py`: Auto-regressively generates 100,000 tokens while measuring the Frobenius norm of the state to verify it does not explode/vanish.
*   Outputs: `results.json`.

### `exp_lr_lowrank/`
**Goal**: The Low-Rank State Ablation.
*   `run_exp.py`: Trains a low-rank HGDM prototype that compresses the per-head state into an r-dimensional latent.
*   `model_lowrank.py`: Defines the low-rank prototype model.
*   Outputs: `results.json`.

## Limitations & Future Work

### State Collapse (The Stuffed Mamba Phenomenon)
We attempted to train HGDM on a Passkey Retrieval task and observed the identical *state collapse* phenomenon documented for Mamba (*Stuffed Mamba*, 2024). Under heavy interference (e.g., thousands of uniform random bytes), the model's state is overwritten before the retrieval query. Because the outer-product state updates write blindly across the memory matrix, high-entropy uniform noise gradually overwrites specific signals. 

This limitation is not unique to HGDM but is mathematically inherent to the write‑over‑everything property of outer‑product state updates. Despite the architecture natively containing learned absolute positional embeddings, the linear gating mechanics still struggle to perfectly isolate scattered patterns across extreme distances. However, because HGDM’s memory complexity is strictly $O(N)$ with a tiny constant, extending the training context or increasing state capacity to mitigate this effect is entirely feasible on a single GPU—a path that remains prohibitively expensive for Transformers. We leave these investigations to future work.

### Low-Rank Prototype
The low-rank prototype is included here as a focused ablation on state compression. It is not yet a full replacement for the main HGDM experiment suite; it is a controlled test of whether smaller latent state can reduce interference and preserve useful recall.

---

## How to Run
Navigate to any experiment folder and run the script:
```bash
cd exp1_enwik8
python run_exp.py
```
Each folder will contain its own clean `results.json` after execution.

For the low-rank prototype:
```bash
cd exp_lr_lowrank
python run_exp.py --steps 200 --seq_len 256 --batch 2 --r 8
```
