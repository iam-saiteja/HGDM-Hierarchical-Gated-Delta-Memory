# HGDM vs. Transformer: Tokenized Chinchilla Arena

This directory contains the subword-tokenized suite for **Hierarchical Gated Delta Memory with Hierarchical Band Search (HGDM-HBS)** and a matched **Baseline Transformer**, designed to compare pure recurrence vs. self-attention under strictly matched Chinchilla compute and data scaling.

---

## Key Principles & Architecture

1. **Strictly Matched Scaling (Chinchilla Optimal)**:
   - Tokens per parameter ratio: $N_{\text{tokens}} = 20 \times N_{\text{params}}$ (Chinchilla compute optimum).
   - Identical layer count ($L=6$), head count ($H=6$), hidden size ($d_{\text{model}}=384$), and SwiGLU intermediate size ($d_{\text{ff}}=1024$).
   - Tied input embedding and output projection matrix.
   - Same tokenizer (`gpt2` 50k BPE) and same streaming data (`HuggingFaceTB/smoltalk`).

2. **Hierarchical Band Search (HBS)**:
   - **Zero Attention / Zero KV-Cache**: 100% pure recurrence ($O(1)$ constant memory).
   - **4 Dyadic Temporal Bands**: Recurrent state is decomposed into powers-of-two temporal scales ($2^0, 2^1, 2^2, 2^3$) with multi-timescale decay anchors ($\tau_0=4, \tau_1=16, \tau_2=64, \tau_3=10,000$).
   - **Dynamic Tournament Router**: Learned routing gates combined with query-key alignment scores dynamically route tokens to appropriate memory bands.
   - **Logarithmic Gradient Reach**: Gradients hop across 1,024 tokens in just 128 steps without vanishing memory decay.

3. **CUDA Graph Pulse Engine**:
   - Fuses all 6 linear projections ($W_q, W_k, W_v, W_{\text{out}}, W_\beta, W_{\text{router}}, W_\delta$) into a single input projection $W_{\text{in}}$.
   - Captures the recurrent step into a static CUDA graph, executing as a single GPU pulse with zero CPU dispatch overhead.
   - Measured generation speed: **505.4 to 1,835.8 tok/s** (a **3.45× speedup** over eager execution).

---

## Directory Layout

```
tokenized/
├── config.py                 # Shared architectural configuration & Chinchilla token calculator
├── model_hgdm.py             # Subword-tokenized HGDM with O(1) memory generation
├── model_transformer.py      # Matched baseline Transformer with KV caching
├── hierarchical_band_memory.py # Hierarchical Band Search (HBS) recurrent mixer
├── cuda_graph.py             # CUDA Graph pulse engine for single-step generation
├── dataset.py                # Streaming tokenized data loader (SmolTalk)
├── train.py                  # Pretraining script (AdamW, Cosine LR, telemetry)
├── train_sft.py              # Prompt-masked Supervised Fine-Tuning pipeline
├── chat_arena.py             # Interactive conversational interface
├── benchmark_inference_full.py # Comprehensive speed, VRAM, and scaling benchmark
├── test_sequence_prompts.py  # Structured sequence completion test (counting, patterns)
├── HBS_THEORY_SPEC.md        # Complete theoretical specification of HBS
├── INFERENCE_KERNEL_TRACKER.md # Roadmap and kernel optimization tracker
├── BENCHMARK_RESULTS.md      # Full pretraining, SFT, and arena empirical log
└── run_full_pipeline.sh      # Automated end-to-end tmux pipeline launcher
```

---

## Empirical Scorecard (RTX 3090 Ti)

| Metric | Matched Transformer | **HGDM with HBS (Ours)** | Winner |
| :--- | :---: | :---: | :---: |
| **Attention Mechanism** | Multi-Head Self-Attention | **Pure Recurrent ($O(1)$)** | **HGDM** |
| **Pretraining Tokens** | 614.1M tokens | **619.5M tokens** | Matched |
| **Conversational SFT Loss** | 2.0148 | **1.9382** | **HGDM (−0.0766)** 🏆 |
| **Conversational Perplexity** | 7.50 | **6.95** | **HGDM (+7.3% lower PPL)** 🏆 |
| **State Memory per User** | Grows $O(T)$ with context | **1,152 KB ($O(1)$ Constant)** | **HGDM (up to 512× smaller)** 🏆 |
| **CUDA Graph Speedup** | N/A (dynamic KV cache) | **3.45× Hardware Acceleration** | **HGDM** |

---

## Usage

### 1. Run Comprehensive Inference Benchmark
```bash
python tokenized/benchmark_inference_full.py
```

### 2. Test Structured Sequential Prompts
```bash
python tokenized/test_sequence_prompts.py
```

### 3. Interactive Conversational Chat
```bash
python tokenized/chat_arena.py
```
