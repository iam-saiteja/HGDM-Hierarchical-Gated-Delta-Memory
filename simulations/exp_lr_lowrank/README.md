# Low-Rank HGDM Prototype

This experiment folder contains a low-rank ablation of HGDM. The goal is to test whether compressing each head's recurrent state into a smaller latent matrix can reduce interference while keeping the model usable for long-context sequence modeling.

## Files
- `model_lowrank.py`: Defines the low-rank prototype model.
- `run_exp.py`: Runs a short synthetic next-token training loop and writes `results.json`.

## Run
From the repository root:
```bash
PYTHONPATH=. python simulations/exp_lr_lowrank/run_exp.py --steps 200 --seq_len 256 --batch 2 --r 8
```

## Output
- `results.json`: Contains the final loss and a short training history.

## Notes
- This is a prototype ablation, not the main HGDM architecture.
- It is intentionally small so it can be rerun quickly on a 3090 Ti.
