Reproducibility checklist

1) Install dependencies

Adjust the PyTorch wheel for your CUDA version. Example for CUDA 11.8:

```bash
pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.2.0+cu118 torchvision==0.15.2+cu118
pip install -r requirements.txt
```

2) Quick verification (does not run full experiments)

- Verify fused vs sequential numerical consistency (if Triton kernel available):

```bash
python verify_fused_vs_sequential.py
```

- Run the low-rank prototype experiment (fast synthetic run):

```bash
cd simulations/exp_lr_lowrank
python run_exp.py --steps 200 --seq_len 256 --batch 2 --r 8
```

Expected verification result:

- `verify_fused_vs_sequential.py` should report small max absolute diffs between sequential and fused paths (on the order of 1e-4 in the current check).

3) Reproducing main experiments

Use the existing `simulations/*/run_exp.py` scripts for full experiments. They require an NVIDIA GPU and Triton for the fused kernel.
The low-rank prototype lives in `simulations/exp_lr_lowrank/` and can be used as a state-compression ablation.

4) Logs & results

- Each experiment writes `results.json` in its folder on completion.
- Save model checkpoints and `results.json` alongside run logs when publishing.

5) Recommended quick checks before publishing

- Run 3 short seed sweeps for `exp1_enwik8` (reduced steps) and report mean±std.
- Run `verify_fused_vs_sequential.py` and include outputs in supplementary.
- Add `requirements.txt` and this `REPRODUCE.md` file to the repository root.
