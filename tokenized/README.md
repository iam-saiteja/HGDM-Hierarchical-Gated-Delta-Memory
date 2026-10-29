# HGDM vs. Transformer: Tokenized Chinchilla Arena

This directory contains the subword-tokenized suite for **Hierarchical Gated Delta Memory (HGDM)** and a matched **Baseline Transformer**, designed to compare recurrence vs. attention under identical compute and data scaling.

---

## Key Principles

1. **Strictly Matched Scaling (Chinchilla Optimal)**:
   - Tokens per parameter ratio: $N_{\text{tokens}} = 20 \times N_{\text{params}}$ (Chinchilla compute optimum).
   - Identical layer count ($L$), head count ($H$), hidden size ($d_{\text{model}}$), and SwiGLU intermediate size ($d_{\text{ff}}$).
   - Tied input embedding and output projection matrix.
   - Same tokenizer (`gpt2` or 32k BPE) and same data stream (`HuggingFaceTB/smoltalk` / `fineweb-edu`).

2. **The Only Variable is the Architecture**:
   - **HGDM**: Gated Delta Matrix Recurrence with continuous ODE variable $\Delta t$ decay, State Normalizer $n_t$, Epistemic Gating, and constant $O(1)$ inference memory (zero KV cache).
   - **Transformer**: Standard Multi-Head Self-Attention with FlashAttention-2 / SDPA and $O(T)$ growing KV-cache.

3. **Multimodal Byte Independence**:
   - The existing byte-level multimodal engine in `ultimate/`, `simulations/`, etc., remains 100% intact and untouched.

---

## Directory Layout

```
tokenized/
├── config.py             # Shared architectural configuration & Chinchilla token calculator
├── model_hgdm.py         # Subword-tokenized HGDM with O(1) memory generation
├── model_transformer.py  # Matched baseline Transformer with KV caching
├── dataset.py            # Streaming tokenized data loader (SmolTalk / FineWeb)
├── train.py              # Matched training loop logging throughput, VRAM, and PPL
├── chat_arena.py         # Interactive conversational interface comparing both models
└── run_remote.sh         # Persistent TMUX launcher for GPU server
```

---

## Scale Tiers

| Tier | Parameters | Training Tokens ($20\times$) | Est. Runtime (RTX 3090 Ti) |
| :--- | :--- | :--- | :--- |
| **`pilot`** | $\sim 15\text{M}$ ($d=384, L=6, H=6$) | $300\text{M}$ tokens | ~2.5 hours total |
| **`standard`** | $\sim 25\text{M}$ ($d=512, L=8, H=8$) | $500\text{M}$ tokens | ~6 to 7 hours |
| **`paper`** | $\sim 120\text{M}$ ($d=768, L=12, H=12$) | $2.4\text{B}$ tokens | ~2.5 days |

---

## Usage

### 1. Training (on Remote GPU via TMUX)
```bash
bash tokenized/run_remote.sh
```
Or run directly:
```bash
python tokenized/train.py --tier pilot --model both --dataset HuggingFaceTB/smoltalk
```

### 2. Conversational Arena
Once trained, launch the interactive side-by-side chat:
```bash
python tokenized/chat_arena.py
```
Or test a single prompt:
```bash
python tokenized/chat_arena.py --prompt "Explain the difference between recurrence and self-attention."
```
