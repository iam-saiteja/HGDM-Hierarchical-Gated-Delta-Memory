# HGDM Experimental Suite

This folder contains the 10 absolute core experiments designed to benchmark the empirical performance of the HGDM architecture against standard Transformers on an RTX 3090 Ti.

Each experiment is organized into its own folder for clean result management.

## Directory Structure

### `exp1_enwik8/`
**Goal**: The headline result. Demonstrates HGDM learns faster and uses less memory than a comparable Transformer.
*   `run_exp.py`: Trains both 120M models for 1000 steps on Enwik8.
*   Outputs: `results.json`, `hgdm_enwik8_120M.pt`, `transformer_enwik8_120M.pt`.

### `exp2_memory/`
**Goal**: The $O(1)$ Memory Validation.
*   `run_exp.py`: Tests sequences from `512` up to `16,384`.
*   Forces the Transformer to use raw math (`FlashAttention` disabled) to expose the $O(N^2)$ quadratic explosion.
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
**Goal**: The Infinite Context Qualitative Test.
*   `run_exp.py`: Loads the trained Enwik8 checkpoint and generates a 2000-byte sequence.
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

## Limitations & Future Work

### State Collapse (The Stuffed Mamba Phenomenon)
We attempted to train HGDM on a Passkey Retrieval task and observed the identical *state collapse* phenomenon documented for Mamba (*Stuffed Mamba*, 2024). Under heavy interference (e.g., thousands of uniform random bytes), the model's state is overwritten before the retrieval query. Because the outer-product state updates write blindly across the memory matrix, high-entropy uniform noise gradually overwrites specific signals. 

This limitation is not unique to HGDM but is mathematically inherent to the write‑over‑everything property of outer‑product state updates without absolute positional embeddings or extensive forgetting curriculums. However, because HGDM’s memory complexity is strictly $O(N)$ with a tiny constant, extending the training context or increasing state capacity to mitigate this effect is entirely feasible on a single GPU—a path that remains prohibitively expensive for Transformers. We leave these investigations to future work.

---

## How to Run
Navigate to any experiment folder and run the script:
```bash
cd exp1_enwik8
python run_exp.py
```
Each folder will contain its own clean `results.json` after execution.
