import torch
from hgdm_ultimate import HGDMUltimate, HGDMConfig

def run_check():
    cfg = HGDMConfig(
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_k=32,
        d_v=32,
        d_ff=256,
        vocab_size=256,
    )
    # Use a sequential fallback for simplicity
    model = HGDMUltimate(cfg, force_sequential=True)

    # Grab the W_alpha weights from the first layer’s mixer
    weight = model.layers[0].mixer.W_alpha.weight
    std = weight.std().item()
    print("[SUCCESS] W_alpha weight std:", std)
    assert std > 0, "W_alpha weights are still zero!"

if __name__ == "__main__":
    run_check()
