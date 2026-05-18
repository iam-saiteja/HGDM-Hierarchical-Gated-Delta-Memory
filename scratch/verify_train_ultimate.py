import torch
import torch.nn.functional as F
import time
from hgdm_ultimate import HGDMUltimate, HGDMConfig
from train_ultimate import train_model


def tiny_dataset(num_tokens=5000):
    # Random bytes dataset for quick sanity check
    return torch.randint(0, 256, (num_tokens,), dtype=torch.long)

def run_test():
    cfg = HGDMConfig(
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_k=32,
        d_v=32,
        d_ff=256,
        vocab_size=256,
    )
    model = HGDMUltimate(cfg, force_sequential=True)
    data = tiny_dataset()
    # Very short training to verify that the warm‑up scheduler and training loop run without errors
    history, total_time = train_model(
        model=model,
        name="tiny_test",
        train_data=data,
        steps=10,
        micro_batch=1,
        accum_steps=2,
        seq_len=64,
        lr=1e-4,
    )
    print("[SUCCESS] train_ultimate ran without errors.")
    print(f"History length: {len(history)}")
    print(f"Total time: {total_time:.2f}s")

if __name__ == "__main__":
    run_test()
