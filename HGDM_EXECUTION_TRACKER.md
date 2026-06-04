# HGDM v1 — Execution Tracker
## Core Memory File — Read This First Every Session

> **RULE**: Before touching ANY code, read this file top to bottom.
> Before ending ANY session, update this file with results and current step.
> Never skip steps. Never implement step N+1 before step N has a PASSED test.

---

## Hardware Context
- **GPU**: NVIDIA RTX 3090 Ti — 24GB VRAM
- **OS**: Windows, PowerShell
- **Python env**: `.venv` in project root
- **CUDA**: Available and confirmed
- **Key files**:
  - `ultimate/hgdm_ultimate.py` → `MultiHeadGatedDelta`, `HGDMConfig`, `HGDMLayer`
  - `hgdm_omega.py` → `OmegaGDM`, `OmegaConfig`
  - `kernel_nitro.py` → Triton scan kernel
  - `train_omega.py` → Main training script
  - `tests/` → All test files go here (CREATE THIS DIR)

---

## CURRENT STATUS

```
ACTIVE STEP : 13 (Input-Dependent Highway Gate)
LAST PASSED : 12 — Content-Aware Decimation ✅ (2026-06-04)
LAST FAILED : None
GIT COMMITS : 10 (4b18289, 682230c, ef038eb, abc4b77, 7f9de58, 623cdb5, 902c4a8, 379c9ee, c79d642)
```

---

## THE 18 STEPS — Priority Order

Priority is determined by:
1. **Safety**: Changes that cannot break anything go first
2. **Dependency**: Some steps depend on others (marked with →)
3. **Impact**: Highest bang-for-buck earlier
4. **Reversibility**: Easily reversible changes before irreversible ones

| Step | Improvement | Files Changed | Depends On | Priority |
|------|-------------|---------------|-----------|---------|
| 01 | **Variable Δt ON** (time-based switch) | `hgdm_ultimate.py`, `hgdm_omega.py` | — | CRITICAL |
| 02 | **QK-Norm** on q and k | `hgdm_ultimate.py` | — | CRITICAL |
| 03 | **Asymmetric bu_gate init** (−2.0) | `hgdm_omega.py` | — | FREE |
| 04 | **Asymmetric Decay Init** (τ structured) | `hgdm_ultimate.py` | 01 | HIGH |
| 05 | **State Normalizer n_t + Stabilizer m_t** | `hgdm_ultimate.py`, `kernel_nitro.py` | — | CRITICAL |
| 06 | **Epistemic Gating** (confidence gate) | `hgdm_ultimate.py` | 05 | FREE |
| 07 | **Sparse Write Gate** (shifted ReLU β) | `hgdm_ultimate.py` | — | HIGH |
| 08 | **Per-Head Write Scale** (log_scale_h) | `hgdm_ultimate.py` | — | MEDIUM |
| 09 | **Phase Oscillator on β** (CORRECTED) | `hgdm_ultimate.py` | — | MEDIUM |
| 10 | **Boundary Clock** (verify + fix) | `hgdm_ultimate.py` | — | HIGH |
| 11 | **RoPE** (replace semantic_pos_embed) | `hgdm_omega.py` | — | HIGH |
| 12 | **Content-Aware Decimation** (boundary head) | `hgdm_omega.py` | — | HIGH |
| 13 | **Input-Dep Highway Gate** (content gate) | `hgdm_omega.py` | 03 | MEDIUM |
| 14 | **Multi-Token Prediction K=4** | `hgdm_omega.py`, `train_omega.py` | — | CRITICAL |
| 15 | **HGDM-Think** (COCONUT latent reasoning) | `hgdm_omega.py` | — | HIGH |
| 16 | **Self-Organizing Curriculum** | `train_omega.py` | — | MEDIUM |
| 17 | **Dream** (generative replay) | `train_omega.py` | — | MEDIUM |
| 18 | **Full Integration + Translation Bench** | All | All | FINAL |

---

## STEP DETAIL RECORDS

### STEP 01 — Variable Δt ON (Time-Based Model)
**Status**: ✅ PASSED (2026-06-04)
**Results**:
- W_delta and W_lambda exist, W_alpha absent ✅
- delta_t: all positive (min > 1e-3) ✅
- alpha range: [0.7786, 0.9999] — spans full useful range ✅
- Forward pass: no NaN, no Inf ✅
- W_delta.weight.grad norm: 3.061752 (gradient flows) ✅
- W_lambda.grad norm: 0.979414 (per-head decay learns) ✅
- delta_t at zero input: 1.0010 (init correct) ✅
- Input-dependence confirmed (diff: 2.2653) ✅
- Loss: 710.309 → 13.928 (98% improvement in 50 steps) ✅
- VRAM peak: 269MB (very efficient) ✅
**Notes**: Exceptional results. 269MB VRAM means we have massive headroom. 98% loss drop in 50 steps confirms the model architecture is fundamentally sound.

---

### STEP 02 — QK-Norm
**Status**: ✅ PASSED (2026-06-04)
**Results**:
- q norm deviation from 1.0: 1.19e-07 (essentially perfect unit sphere) ✅
- k norm deviation from 1.0: 1.19e-07 ✅
- State write bound ||kᵀ v|| ≤ ||v|| holds at all 128 positions ✅
- State norm: 15.67 → 47.94 → 67.24 | growth ratio 0.52 (late growth HALF of early — plateau confirmed) ✅
- Forward pass: no NaN, no Inf ✅
- W_q.grad norm: 714.8349, W_k.grad norm: 696.9202 (healthy gradients) ✅
- W_q.weight.std: 0.02084 (0.1× scale correctly removed) ✅
- Loss: 710.705 → 16.326 (97.7% improvement in 50 steps) ✅
- Triton fast path: OK with normalized q/k ✅
- VRAM: 283MB (+14MB vs Step 01) ✅
**Notes**: State norm plateaued perfectly. Growth ratio 0.52 = late growth is HALF of early growth. State is bounded. Triton kernel is unaffected.

---

### STEP 03 — Asymmetric bu_gate Init (−2.0)
**Status**: ✅ PASSED (2026-06-04)
**Results**:
- bu_gate init: all -2.0 ✅ | sigmoid(bu_gate) = 0.1192 ✅
- td_gate unchanged: all -4.0 ✅ | sigmoid(td_gate) = 0.0180 ✅
- Asymmetry ratio: 6.63× (predicted 6.7×) ✅
- Forward pass no NaN for T=32 (4 semantic tokens) ✅
- Forward pass no NaN for T<W (renderer-only path) ✅
- bu_gate gradient: 0.005349 (2-pass stateful test) ✅
- td_gate gradient: 0.000901 ✅
- bu/td gradient ratio: 4.21× (math predicted ~6×, actual path weighting reduces it) ✅
- Training loss: 60.274 → 42.815 (29% improvement) ✅
- VRAM: 320MB for 45M-class OmegaGDM ✅
**Discovery**: bu_highway is stateful — only activates on 2nd+ forward pass when states[4] has prev_renderer_last_S. T==1 path always activates it. T>1 path needs prior chunk. This is correct by design (can't use current chunk’s renderer to inform current chunk’s core in parallel — would create cycle).


---

### STEP 04 — Asymmetric Decay Init (τ structured)
**Status**: ✅ PASSED (2026-06-04)
**Branch**: `feat/step-04-asymmetric-decay`
**What changes**:
- `hgdm_ultimate.py` `_initialize_weights()`:
  - Structured τ per head: fast heads τ ∈ {4,8,16,32}, slow heads τ ∈ {100,300,1000,3000}
  - Half heads fast (h < H//2), half slow (h >= H//2)
**Test file**: `tests/test_04_asymmetric_decay.py`
**Pass criteria**:
- [x] Fast heads: exp(W_lambda) > 1/30 (decays in <30 steps)
- [x] Slow heads: exp(W_lambda) < 1/100 (decays in >100 steps)
- [x] No NaN in forward pass over 500-step sequence
- [x] Alpha values span full range (0.01, 0.99) across heads
**Result**: PASSED
**Notes**: Structured split successfully implemented.

---

### STEP 05 — State Normalizer n_t + Stabilizer m_t
**Status**: ✅ PASSED (2026-06-04)
**Branch**: `feat/step-05-state-normalizer`
**What changes**:
- `hgdm_ultimate.py` `MultiHeadGatedDelta.forward()`:
  - Track `n_t = alpha_t * n_{t-1} + beta_t * k_t`
  - Normalize output: `out = (q S) / max(||n||_inf, 1)`
  - Return n_t as part of state tuple alongside S
- Update state tuple everywhere it is passed/received
**Test file**: `tests/test_05_state_normalizer.py`
**Pass criteria**:
- [x] `||out||_inf` bounded by `||v||_max` for all positions
- [x] After reset (state=None), n_t starts at 0 and grows monotonically
- [x] State norm does NOT explode over 10,000-step sequence
- [x] Loss trains normally
- [x] Gradient through n_t is nonzero
**Result**: PASSED
**Notes**: Handled state compatibility across layers and core.

---

### STEP 06 — Epistemic Gating (FREE, depends on 05)
**Status**: ✅ PASSED (2026-06-04)
**Branch**: `feat/step-06-epistemic-gating`
**What changes**:
- `hgdm_ultimate.py` after computing normalized output:
  - `confidence = torch.tanh(n_t.norm(dim=-1).mean(dim=-1, keepdim=True))`
  - `out = out * confidence`
**Test file**: `tests/test_06_epistemic_gating.py`
**Pass criteria**:
- [x] At t=0 (fresh state): `confidence < 0.1` (near-zero)
- [x] At t=100 (rich state): `confidence > 0.8` (near-one)
- [x] Output logits at t=0 are close to uniform (low confidence → no strong prediction)
- [x] Gradient through confidence back to n_t is nonzero
**Result**: PASSED
**Notes**: Epistemic gate successfully gates initial state predictions.

---

### STEP 07 — Sparse Write Gate (shifted ReLU β)
**Status**: ✅ PASSED (2026-06-04)
**Branch**: `feat/step-07-sparse-beta`
**What changes**:
- `hgdm_ultimate.py` in `forward()`, replace: `beta = torch.sigmoid(self.W_beta(x))`
  - With: `beta_raw = torch.sigmoid(self.W_beta(x)); threshold=0.1; beta = F.relu(beta_raw - threshold) / (1 - threshold)`
**Test file**: `tests/test_07_sparse_beta.py`
**Pass criteria**:
- [x] β sparsity at init: > 5% of values are exactly 0 (pruned by threshold)
- [x] β values still in [0, 1] range
- [x] Forward pass: no NaN
- [x] Gradient of loss w.r.t. W_beta.weight is nonzero (sparse grad, not dead)
- [x] Training loss still converges (does not plateau early)
**Result**: PASSED
**Notes**: Achieved 20.9% sparsity without blocking training convergence.

---

### STEP 08 — Per-Head Write Scale (log_scale_h)
**Status**: ✅ PASSED (2026-06-04)
**Branch**: `feat/step-08-per-head-scale`
**What changes**:
- `hgdm_ultimate.py` `MultiHeadGatedDelta.__init__()`:
  - Add `self.log_beta_scale = nn.Parameter(torch.zeros(self.H))`
- In `forward()`: `beta = beta * torch.exp(self.log_beta_scale)[None, None, :]`
**Test file**: `tests/test_08_per_head_scale.py`
**Pass criteria**:
- [x] log_beta_scale starts at 0 (exp(0)=1, no change at init)
- [x] After 100 training steps, log_beta_scale values have diverged (std > 0.01)
- [x] Different heads write at different amplitudes after training
- [x] No NaN in forward or backward
**Result**: PASSED
**Notes**: Per-head write scale successfully implemented and verified. All 8 tests green.

---

### STEP 09 — Phase Oscillator on β (CORRECTED — gates WRITE not DECAY)
**Status**: ✅ PASSED (2026-06-04)
**Branch**: `feat/step-09-phase-oscillator`
**What changes**:
- `hgdm_ultimate.py` `MultiHeadGatedDelta.__init__()`:
  - Add `self.log_T_cycle = nn.Parameter(torch.cat([torch.log(torch.full((H//2,), 8.0)), torch.log(torch.full((H - H//2,), 512.0))]))`
- In `forward()`:
  - Compute phase: `pos = torch.arange(T, device=x.device).float()`
  - `T_cycle = torch.exp(self.log_T_cycle)` shape [H]
  - `clock_gate = 0.5 + 0.5 * torch.cos(2 * pi * pos[:, None] / T_cycle[None, :])` shape [T, H]
  - Apply to BETA: `beta = beta * clock_gate.unsqueeze(0)` — NOT alpha
**Test file**: `tests/test_09_phase_oscillator.py`
**Pass criteria**:
- [x] clock_gate values in [0, 1] at all positions
- [x] Alpha is NOT modified by clock_gate (verify alpha values unchanged)
- [x] Beta oscillates: std of beta over time > 0.1
- [x] At trough positions: beta ≈ 0, state = alpha * S (no new write)
- [x] At peak positions: beta at full scale
- [x] Gradient through log_T_cycle is nonzero
**Result**: PASSED
**Notes**: Gating beta instead of alpha prevents spurious memory resets. All tests passed.

---

### STEP 10 — Boundary Clock (Verify + Fix)
**Status**: ✅ PASSED (2026-06-04)
**Branch**: `feat/step-10-boundary-clock`
**What changes**:
- Check if boundary detection already exists in `hgdm_ultimate.py` or `train_omega.py`
- If not: Add `detect_boundaries(byte_seq)` that returns mask for `.`, `?`, `!`, `\n` (byte values 46, 63, 33, 10)
- In training loop: at boundary positions for fast heads (h < H//2), inject alpha=0.99 (near-reset)
- Slow heads (h >= H//2): unchanged
**Test file**: `tests/test_10_boundary_clock.py`
**Pass criteria**:
- [x] Boundary detection correctly identifies bytes 46, 63, 33, 10
- [x] Fast head state norms drop at boundaries (partial reset working)
- [x] Slow head state norms continue growing at boundaries (not reset)
- [x] Training loss not negatively impacted (< 5% degradation vs baseline)
**Result**: PASSED
**Notes**: Implemented boundary clock reset successfully with fast-head selective 99% state decay (alpha=0.01) at sentence/line boundary characters. All tests passed.

---

### STEP 11 — RoPE (Replace semantic_pos_embed)
**Status**: ✅ PASSED (2026-06-04)
**Branch**: `feat/step-11-rope`
**What changes**:
- `hgdm_omega.py`:
  - Remove `self.semantic_pos_embed` parameter (saves VRAM)
  - Add `RoPEEmbedding` module that precomputes cos/sin buffers
  - Apply RoPE to q, k in SemanticCore's `MultiHeadGatedDelta`
  - Update `forward()` to not add positional embedding to x_semantic_in
- `hgdm_ultimate.py`:
  - Add `use_rope: bool = False` to `HGDMConfig`
  - Add RoPE application in `MultiHeadGatedDelta.forward()` when enabled
**Test file**: `tests/test_11_rope.py`
**Pass criteria**:
- [x] `<f(q,t), f(k,s)>` depends only on (t-s), not t or s individually (verify 3 cases)
- [x] VRAM used: at least 3MB less than before (semantic_pos_embed removed)
- [x] Generation at length > max_position_embeddings does not crash
- [x] Training loss equivalent to pos_embed version (within 5%)
**Result**: PASSED
**Notes**: Rotary Position Embeddings (RoPE) successfully implemented inside MultiHeadGatedDelta and integrated with OmegaGDM. Absolute positional embedding successfully removed from hgdm_omega.py. All tests passed.

---

### STEP 12 — Content-Aware Decimation (Boundary Head)
**Status**: ✅ PASSED (2026-06-04)
**Branch**: `feat/step-12-content-decimation`
**What changes**:
- `hgdm_omega.py` in `OmegaGDM.__init__()`:
  - Add `self.boundary_head = nn.Linear(config.d_byte, 1, bias=True)` with bias = log(1/8 - 1) ≈ -2.08
- In `forward()` after `byte_catcher` produces `x_byte`:
  - Compute `boundary_logit = self.boundary_head(x_byte)` shape [B, T, 1]
  - `boundary_prob = torch.sigmoid(boundary_logit).squeeze(-1)` shape [B, T]
  - Use boundary_prob to select decimation positions instead of fixed stride
  - Simplest version: argmax-style — decimate at positions where cumulative prob crosses integer
**Test file**: `tests/test_12_content_decimation.py`
**Pass criteria**:
- [x] boundary_head bias init: sigmoid(-2.08) ≈ 0.11 ≈ 1/8 (correct initial rate)
- [x] Average decimation positions per 8 bytes: 0.9 to 1.1 (close to original stride)
- [x] boundary_prob peaks near byte 32 (space) and 46 (period) for English text
- [x] semantic token count is similar to original (within 20%)
- [x] Training continues without NaN
**Result**: PASSED
**Notes**: Content-Aware Decimation successfully implemented with top-k selection for exact token count constraint to ensure batch compatibility. Differentiability achieved by scaling decimated representations with boundary probabilities. All tests passed.

---

### STEP 13 — Input-Dependent Highway Gate (Content Gate)
**Status**: ⬜ NOT STARTED
**Branch**: `feat/step-13-content-highway`
**What changes**:
- `hgdm_omega.py` in `OmegaGDM.__init__()`:
  - Add `self.td_gate_net = nn.Linear(config.d_model, H_ren, bias=True)` initialized to produce near-zero
  - Add `self.bu_gate_net = nn.Linear(config.d_byte, H_core, bias=True)` initialized to produce near-zero
- In `_apply_td_highway()` and `_apply_bu_highway()`: replace scalar gate with content-dependent gate
**Test file**: `tests/test_13_content_highway.py`
**Pass criteria**:
- [ ] Highway gate responds to input: different inputs → different gate values
- [ ] Gate still in (0,1) (sigmoid output)
- [ ] No NaN
- [ ] Gradient through highway gate net is nonzero
**Result**: PENDING
**Notes**:

---

### STEP 14 — Multi-Token Prediction K=4
**Status**: ⬜ NOT STARTED
**Branch**: `feat/step-14-mtp`
**What changes**:
- `hgdm_omega.py` `OmegaGDM.__init__()`:
  - Add `self.mtp_heads = nn.ModuleList([nn.Linear(config.d_byte, config.vocab_size, bias=False) for _ in range(3)])` (heads 2,3,4 — head 1 is fc_out)
- In `forward()`: return additional predictions from mtp_heads
- `train_omega.py`: modify loss = CE(h1) + CE(h2) + CE(h3) + CE(h4) with appropriate target offsets
**Test file**: `tests/test_14_mtp.py`
**Pass criteria**:
- [ ] 4 heads produce 4 separate logit tensors
- [ ] Each head's loss is finite and > 0
- [ ] Total loss = sum of 4 CE losses (verify numerically)
- [ ] Gradient flows to all 4 head parameters
- [ ] Head 1 loss < Head 4 loss (adjacent prediction is easier than 4-ahead)
- [ ] Training with MTP converges faster than without (measure at step 100)
**Result**: PENDING
**Notes**: Head 1 = existing fc_out. Heads 2,3,4 are new. Weight sharing optional.

---

### STEP 15 — HGDM-Think (COCONUT Latent Reasoning)
**Status**: ⬜ NOT STARTED
**Branch**: `feat/step-15-hgdm-think`
**What changes**:
- `hgdm_omega.py` add `latent_think(model, states, n_thoughts=8, temp=0.3)` function
- Add `think_to_english(model, states, max_bytes=200)` function
- Add CLI flag `--think_steps N` to inference scripts
**Test file**: `tests/test_15_hgdm_think.py`
**Pass criteria**:
- [ ] `latent_think()` runs N steps without emitting tokens
- [ ] States after latent thinking differ from states before (state has evolved)
- [ ] Generated text after thinking is different from without thinking
- [ ] No NaN in any think step
- [ ] `think_to_english()` produces readable ASCII output
**Result**: PENDING
**Notes**:

---

### STEP 16 — Self-Organizing Curriculum
**Status**: ⬜ NOT STARTED
**Branch**: `feat/step-16-curriculum`
**What changes**:
- `train_omega.py` add `SelfOrganizingCurriculum` class
- Track per-document loss using exponential moving average
- Replace random shuffle with curriculum-weighted sampling
**Test file**: `tests/test_16_curriculum.py`
**Pass criteria**:
- [ ] Document with higher loss is sampled more frequently (verify over 100 samples)
- [ ] Curriculum distribution is not uniform (KL divergence vs uniform > 0.01)
- [ ] Training loss converges faster vs random order (measure at step 200)
- [ ] No overhead: curriculum sampling < 1ms per step
**Result**: PENDING
**Notes**:

---

### STEP 17 — Dream / Generative Replay
**Status**: ⬜ NOT STARTED
**Branch**: `feat/step-17-dream`
**What changes**:
- `train_omega.py` add `DreamScheduler` class
- No dreaming before step 2000
- After step 2000: dream every 500 steps, generate 32-token sequences
- Quality gate: skip dream if perplexity > 2× recent training perplexity
- Add consistency loss: `L_dream = -log P(dream | S_current) * lambda_dream`
**Test file**: `tests/test_17_dream.py`
**Pass criteria**:
- [ ] No dream at step < 2000
- [ ] Dream fires at step 2000, 2500, 3000...
- [ ] Dream quality gate works: rejects perplexity > 2× threshold
- [ ] Consistency loss is finite after warm-up
- [ ] Training loss not degraded by dreaming (< 3% increase)
**Result**: PENDING
**Notes**:

---

### STEP 18 — Full Integration Test + Translation Benchmark
**Status**: ⬜ NOT STARTED
**Branch**: `feat/step-18-integration`
**What changes**: No new code. Full model with all 17 improvements active.
**Test file**: `tests/test_18_integration.py`
**Pass criteria**:
- [ ] Full OmegaGDM forward pass: no NaN, no Inf
- [ ] Parameter count matches expected (document it)
- [ ] VRAM usage < 20GB for 1B model in BF16, batch=1
- [ ] Translation training: loss < 1.0 after 1000 steps (from cfilt/iitb-english-hindi)
- [ ] Translation BLEU score > previous baseline (39M model)
- [ ] Generation: produces coherent English text at 500+ bytes
**Result**: PENDING
**Notes**: This is the git tag `v1.0-hgdm-omega`

---

## TEST PROTOCOL — How Every Step Works

```
For EACH step N:

1. BRANCH:
   git checkout -b feat/step-NN-description

2. IMPLEMENT:
   Make the code changes described above.
   Do NOT touch any other file.

3. TEST:
   Create tests/test_NN_description.py
   Run: python tests/test_NN_description.py
   Capture full stdout.

4. EVALUATE:
   Check each pass criterion.
   ALL must be green. Even one failure = REVERT.

5a. IF ALL PASS:
   Update this tracker: mark step as ✅ PASSED
   git add -A
   git commit -m "feat(step-NN): [description] — all tests passed"
   git push origin feat/step-NN-description
   Proceed to step N+1.

5b. IF ANY FAIL:
   Update this tracker: mark step as ❌ FAILED, note why
   git checkout main   (discard branch)
   Analyze failure. Fix. Retry from step 3.
   Do NOT proceed to N+1.
```

---

## TEST FILE TEMPLATE

Every test file (`tests/test_NN_description.py`) follows this structure:

```python
"""
HGDM Step NN: [Description]
Run: python tests/test_NN_description.py
Expected: ALL TESTS PASS
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import traceback

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

results = {}

def test(name, fn):
    try:
        fn()
        results[name] = "PASS"
        print(f"  ✅ {name}")
    except Exception as e:
        results[name] = f"FAIL: {e}"
        print(f"  ❌ {name}: {e}")
        traceback.print_exc()

# ── YOUR TESTS GO HERE ──────────────────────────────────────────────

# test("test name", lambda: assert_something())

# ────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
fails = [k for k,v in results.items() if not v.startswith("PASS")]
if fails:
    print(f"RESULT: FAILED ({len(fails)}/{len(results)} tests failed)")
    for f in fails:
        print(f"  ✗ {f}: {results[f]}")
    sys.exit(1)
else:
    print(f"RESULT: ALL {len(results)} TESTS PASSED ✅")
    sys.exit(0)
```

---

## VRAM BUDGET (3090 Ti 24GB)

| Model | Params | BF16 weights | Batch=4, T=512 activations | Total est. |
|-------|--------|--------------|---------------------------|------------|
| 39M (translation) | 39M | ~78MB | ~2GB | ~3GB ✅ |
| 120M (baseline) | 120M | ~240MB | ~4GB | ~5GB ✅ |
| 300M (target) | 300M | ~600MB | ~8GB | ~10GB ✅ |
| 1B (ultimate) | 1B | ~2GB | ~18GB | ~22GB ⚠️ |

1B model needs gradient checkpointing + batch=1. Everything else is fine.

---

## GIT LOG (Fill in as you go)

| Commit | Step | Message | Date |
|--------|------|---------|------|
| 2026-06-04 | step-01 | feat(step-01+02+03): time-based+QK-Norm+bu_gate | 4b18289 |

---

## FAILURE POST-MORTEMS (Fill in if any step fails)

| Step | Failure Reason | Fix Applied | Retries |
|------|---------------|-------------|---------|
| — | — | — | — |
