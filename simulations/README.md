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

### Exp 11: Passkey Retrieval & State Collapse (The Stuffed Mamba Phenomenon)
- **Goal**: Evaluate the effective context window via synthetic single-byte retrieval.
- **Method**: Curriculum training from 512 to 16,384 sequence lengths using the pre-trained Enwik8 gating mechanisms.
- **Result (The Stuffed Mamba Limitation)**: While HGDM successfully maintains physical state stability over 100k tokens (Exp 10), it suffers from the identical *state collapse* phenomenon documented in recent SSM literature (e.g., *Stuffed Mamba*, 2024). Because the outer-product state updates write blindly across the memory matrix, high-entropy uniform noise gradually overwrites the passkey signal. Without absolute positional embeddings or multi-day extensive forgetting curriculums, pure linear associative memories struggle with synthetic retrieval over extreme distances. This confirms HGDM's theoretical isomorphism to Mamba-class architectures in both strengths (O(1) inference) and boundaries (state interference).
*   Outputs: `results.json`.

---

## How to Run
Navigate to any experiment folder and run the script:
```bash
cd exp1_enwik8
python run_exp.py
```
Each folder will contain its own clean `results.json` after execution.
