<div align="center">
  <h1>🌌 HTSPC: Hierarchical Gated Delta Memory (HGDM)</h1>
  <p><strong>A Constant-Memory, Attention-Free Sequence Model for Infinite Context Processing</strong></p>
</div>

<br>

> [!IMPORTANT]
> **HTSPC-H3 (HGDM-Ultimate)** represents a fundamental breakthrough in sequence modeling. By replacing the $O(N^2)$ Self-Attention mechanism with a mathematically sound, multi-scale **Gated Delta Rule** computed via a highly optimized Triton kernel, HGDM achieves **$O(1)$ memory during inference** and strictly linear $O(N)$ scaling during training. 

---

## 📑 Table of Contents
1. [The Attention Bottleneck](#1-the-attention-bottleneck)
2. [The HGDM Architecture](#2-the-hgdm-architecture)
   - [The Gated Delta Rule](#the-gated-delta-rule)
   - [Multi-Scale $\tau$ Initialization (The Secret Sauce)](#multi-scale-tau-initialization)
   - [V7 Nitro Triton Kernel](#v7-nitro-triton-kernel)
3. [Empirical Validations (The Faceoff)](#3-empirical-validations)
   - [Exp 1: Context Memory Scaling](#exp-1-context-memory-scaling)
   - [Exp 2: Hardware Stress Test](#exp-2-hardware-stress-test)
   - [Exp 3: Multi-Scale Ablation](#exp-3-multi-scale-ablation)
4. [Installation & Setup](#4-installation--setup)
5. [Training & Evaluation](#5-training--evaluation)
6. [Future Work: Scaling Laws](#6-future-work)
7. [License & Citation](#7-license)

---

## 1. The Attention Bottleneck

Standard Transformers process sequences by comparing every token to every previous token. This operation, known as **Self-Attention**, requires computing an $N \times N$ attention matrix. 
- **Training Time/Memory**: Scales quadratically $O(N^2)$. While modern software optimizations like FlashAttention obscure this bottleneck for short sequences, it remains fundamentally insurmountable for millions of tokens.
- **Inference Time/Memory**: Scales linearly $O(N)$. The KV-Cache grows indefinitely with every generated token, eventually causing Out-Of-Memory (OOM) errors and drastically slowing down generation speed.

**The HTSPC Solution:** We must abandon attention entirely. Instead of comparing the current token to the entire history, we compress the history into a fixed-size mathematical state, updating it recursively at every time step.

---

## 2. The HGDM Architecture

The **Hierarchical Gated Delta Memory (HGDM)** architecture is built on three core pillars:

### The Gated Delta Rule
Instead of calculating softmax over the past, HGDM maintains a hidden state matrix $S_t$. At each step $t$, the state is decayed by a forget gate $\alpha_t$ and updated by a new input delta controlled by an input gate $\beta_t$.

$$ S_t = \alpha_t \odot S_{t-1} + \beta_t \odot (K_t \otimes V_t) $$
$$ O_t = (Q_t \otimes S_t) \odot \text{OutGate}_t $$

This formulation ensures that the size of $S_t$ remains perfectly constant ($O(1)$), regardless of how many tokens the model has processed.

### Multi-Scale $\tau$ Initialization
A single recurrent state struggles to remember both short-term syntax and long-term context simultaneously. HGDM solves this by splitting its attention heads across different timescales. We initialize the bias of the $\alpha$ forget gates using a hierarchical array of $\tau$ (tau) values:

```python
base_taus = [4.0, 30.0, 200.0, 1200.0, 8000.0]
alpha_target = math.exp(-1.0 / tau)
bias_val = math.log(alpha_target / (1.0 - alpha_target + 1e-8))
```
* **Head 0 ($\tau=4$)**: Forgets quickly; specializes in local syntax and grammar.
* **Head 4 ($\tau=8000$)**: Remembers indefinitely; specializes in long-range document context.

### V7 Nitro Triton Kernel
Recurrent models are notoriously slow to train because they cannot be easily parallelized across the time dimension. We engineered the **V7 Nitro Engine**, a custom `triton` kernel that chunk-parallelizes the recurrent scan. It computes the intra-chunk interactions in parallel using standard matrix multiplication, and only passes the $S_t$ state sequentially between chunks.

---

## 3. Empirical Validations

We conducted extensive physical hardware benchmarking pitting a 120M parameter PyTorch Transformer against the 120M parameter HGDM on a single 24GB consumer GPU.

### Exp 1: Context Memory Scaling (Raw Math)
We disabled `FlashAttention` in PyTorch to expose the raw mathematical complexity of both models.

| Sequence Length | Transformer Peak VRAM | HGDM Peak VRAM | Status |
| :--- | :--- | :--- | :--- |
| **512** | 2,719 MB | 2,657 MB | Both Survived |
| **2048** | 8,490 MB | 3,691 MB | Transformer Exploding |
| **4096** | **OOM (Failed)** | 5,340 MB | Transformer Dead |
| **8192** | **OOM (Failed)** | 8,678 MB | HGDM Linear Scaling |

**Conclusion**: Transformers fundamentally crash on large contexts without extreme software workarounds. HGDM scales gracefully and linearly.

### Exp 2: Hardware Stress Test
We pushed the HGDM to extreme limits to see how far linear scaling could go on a consumer GPU.
- **16,384 Tokens**: Survived effortlessly (14.9 GB VRAM).
- **32k+ Tokens**: Reached OOM. 
> *Note: While HGDM has $O(1)$ recurrent memory during inference, PyTorch Backpropagation Through Time (BPTT) still caches the inputs to Linear layers. To train on 131k context, standard Gradient Checkpointing must be enabled.*

### Exp 3: Multi-Scale Ablation
We trained three variants of the 120M model on Enwik8/TinyShakespeare for 2,000 steps to prove the gating theory.

1. **Learned (No Bias)**: Settled at 2.69 BPB.
2. **Flat ($\tau=200$)**: Settled at 2.54 BPB.
3. **Full (Multi-scale $\tau$)**: Achieved the lowest loss consistently, proving that spacing out memory decay rates across heads is mathematically superior.

---

## 4. Installation & Setup

> [!WARNING]
> This codebase relies heavily on custom CUDA/Triton kernels. You must be on a Linux environment with a modern NVIDIA GPU and PyTorch compiled with CUDA support.

```bash
# Clone the repository
git clone https://github.com/iam-saiteja/HTPSC.git
cd HTPSC

# Create a fresh environment
python -m venv venv
source venv/bin/activate

# Install dependencies (ensure PyTorch matches your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install triton matplotlib numpy
```

---

## 5. Training & Evaluation

The codebase is highly modularized for experimentation.

### Running the Head-to-Head Faceoff
To execute the raw mathematical comparison between HGDM and standard Transformers:
```bash
cd simulations/exp_faceoff
python run_faceoff.py
```

### Running the Enwik8 Production Training
To train the ultimate model on the Enwik8 dataset:
```bash
python benchmarks/v4/train_ultimate.py
```
This script handles dataset downloading, chunking, mixed-precision training (BF16), and outputs a highly detailed markdown report evaluating BPB and Perplexity.

### Training the Math Module
```bash
python math/train_math.py
```

---

## 6. Future Work

The current 120M implementation proves the viability of the architecture. The next phase involves scaling the system to the 1.1B parameter regime (Titan scale) to study:
1. **Emergent Reasoning**: Can an attention-free model perform multi-step mathematical chain-of-thought?
2. **Infinite Context Retrieval**: Evaluating "Needle In A Haystack" accuracy at 1M+ tokens.
3. **Instruction Tuning**: Shifting from pure byte-level autoregressive modeling to formatted chat interactions.

---

## 7. License
This project is proprietary research. All rights reserved. Do not distribute without explicit permission from the authors.
