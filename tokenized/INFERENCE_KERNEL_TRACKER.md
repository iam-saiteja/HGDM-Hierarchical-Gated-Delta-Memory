# HGDM Fused Step Kernel & Hierarchical Architecture Tracker

**Branch**: `feat/hierarchical-fused-step-kernel`  
**Status**: In Progress  
**Objective**: 
1. Fix the inference throughput bottleneck by developing a lightweight **Triton Single-Step Gated (TSG) Fused Kernel** for $T=1$ autoregressive generation.
2. Maintain persistent recurrent state ($S \in \mathbb{R}^{d_k \times d_v}$ and $n \in \mathbb{R}^{d_k}$) in GPU registers/SRAM, eliminating Python dispatch overhead and boosting inference speed from ~200 tok/s to >1,000 tok/s.
3. Prepare the mathematical foundation for **Logarithmic Dyadic States** (Hierarchical binary state tree hopping across powers of 2: $t - 2^k$) to eliminate needle-overwriting in pure $O(1)$ recurrence without attention matrices.

---

## 1. Diagnosis of Existing Inference Bug in `main`

### Root Cause:
* The existing Triton kernel in `kernel_nitro.py` was designed exclusively for **chunked forward passes** where `T % chunk_size == 0` ($T \ge 32$).
* During conversational chat and autoregressive inference (`tokenized/chat_arena.py` and `tokenized/run_comprehensive_arena.py`), token generation runs one token at a time ($T = 1$).
* Because $1 \pmod{32} \neq 0$, the Triton engine **refused to run** and silently dropped back to an **unrolled Python loop in PyTorch**:
  ```python
  for t in range(T):
      delta = torch.einsum('bhk,bhd->bhkd', k[:, t], v[:, t])
      S = alpha[:, t, :, None, None] * S + beta[:, t, :, None, None] * delta
      out_t = torch.einsum('bhkd,bhk->bhd', S, q[:, t])
      ...
  ```
* This launched **36 separate PyTorch CUDA kernel overheads per token** across 6 layers, limiting generation speed to ~100–300 tok/s.
* Furthermore, any prompt with sequence length not divisible by 32 (e.g., Prompt 1 with 39 tokens) was forced into PyTorch fallback mode during prompt ingestion.

---

## 2. Architecture of the TSG Fused Step Kernel

For a single step $T = 1$:
* **Grid**: $(B, H)$ — exactly one thread block per head per batch element.
* **Block Dimensions**: $D_K = 64, D_V = 64$.
* **Fused Register Operations**:
  1. Load $S_{\text{prev}} \in \mathbb{R}^{D_K \times D_V}$ and $n_{\text{prev}} \in \mathbb{R}^{D_K}$ into fast SM registers.
  2. Compute outer product update: $\Delta = k^\top v \in \mathbb{R}^{D_K \times D_V}$.
  3. Update state in registers: $S_{\text{new}} = \alpha S_{\text{prev}} + \beta \Delta$.
  4. Perform query readout: $o = q \cdot S_{\text{new}} \in \mathbb{R}^{D_V}$.
  5. Update normalizer vector: $n_{\text{new}} = \alpha n_{\text{prev}} + \beta k$.
  6. Compute epistemic confidence: $\text{conf} = \tanh(\|n_{\text{new}}\|_2) / \max(1.0, \|n_{\text{new}}\|_\infty)$.
  7. Apply output gating: $o_{\text{final}} = o \cdot \text{conf} \cdot g$.
  8. Store $S_{\text{new}}, n_{\text{new}}, o_{\text{final}}$ back in a single coalesced memory transaction.

* **Target Speed**: < 20 microseconds per layer on RTX 3090 Ti $\implies$ **1,200+ tok/s generation throughput**.

---

## 3. Implementation Roadmap

- [ ] **Phase 1: Triton Single-Step Kernel (`fused_nitro_step`)**
  - Implement `@triton.jit def _step_fwd_kernel` in `kernel_nitro.py`.
  - Provide python wrapper `fused_nitro_step(q, k, v, alpha, beta, S_prev, n_prev)`.
  - Add numerical verification test comparing PyTorch reference vs Triton step kernel.
- [ ] **Phase 2: Integration into `TokenizedHGDM`**
  - Modify `TokenizedHGDM.forward` and `TokenizedHGDM.generate` to call `fused_nitro_step` whenever $T = 1$.
  - Benchmark generation throughput in `chat_arena.py` on remote RTX 3090 Ti.
- [ ] **Phase 3: Verification & Merge into `main`**
  - Ensure all tests pass with zero divergence.
  - Merge into `main` with updated documentation.
- [ ] **Phase 4: Hierarchical Binary State Tree (Logarithmic $O(\log T)$ Dyadic States)**
  - Implement power-of-two dyadic skip connections ($2^0, 2^1, 2^2, \dots, 2^k$).
  - Eliminate needle-overwriting over long contexts in pure recurrence.
