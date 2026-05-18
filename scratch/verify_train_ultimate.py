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
    from train_ultimate import TransformerTied
    tied_model = TransformerTied(d_model=128, n_layers=2, n_heads=4, d_ff=256, vocab_size=256, max_seq_len=64)
    data = tiny_dataset()
    
    # Very short training for HGDM
    history, total_time = train_model(
        model=model,
        name="tiny_test_hgdm",
        train_data=data,
        steps=10,
        micro_batch=1,
        accum_steps=2,
        seq_len=64,
        lr=1e-4,
    )
    
    # Very short training for Tied Transformer
    history_tied, total_time_tied = train_model(
        model=tied_model,
        name="tiny_test_tied_transformer",
        train_data=data,
        steps=10,
        micro_batch=1,
        accum_steps=2,
        seq_len=64,
        lr=1e-4,
    )
    print("[SUCCESS] train_ultimate ran both HGDM and Tied Transformer without errors.")
    print(f"HGDM time: {total_time:.2f}s | Tied Transformer time: {total_time_tied:.2f}s")

if __name__ == "__main__":
    run_test()
