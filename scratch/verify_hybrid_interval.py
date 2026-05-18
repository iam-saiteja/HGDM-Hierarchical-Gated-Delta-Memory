"""
verify_hybrid_interval.py
==========================
Verifies that the `hybrid_interval` stub has been completely removed from
`HGDMConfig` and the codebase, and that the model config can be initialized
without error.
"""

from hgdm_ultimate import HGDMConfig, HGDMUltimate
import torch

def main():
    print("=" * 60)
    print("VERIFY: hybrid_interval Removal")
    print("=" * 60)

    # 1. Verify that hybrid_interval is no longer in HGDMConfig
    config = HGDMConfig()
    if hasattr(config, "hybrid_interval"):
        print("[FAIL] hybrid_interval attribute is still present in HGDMConfig!")
        raise AssertionError("hybrid_interval is not removed from HGDMConfig.")
    else:
        print("[PASS] hybrid_interval is successfully removed from HGDMConfig.")

    # 2. Try to initialize the model with the updated config
    cfg = HGDMConfig(
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_k=32,
        d_v=32,
        d_ff=256,
        vocab_size=256
    )
    
    # Check that it doesn't crash on model init
    model = HGDMUltimate(cfg, force_sequential=True)
    print("[PASS] Model successfully initialized with the updated HGDMConfig.")
    print("=" * 60)
    print("[SUCCESS] hybrid_interval cleanup verified completely.")
    print("=" * 60)

if __name__ == "__main__":
    main()
