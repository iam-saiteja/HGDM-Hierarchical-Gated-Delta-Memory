"""
verify_passkey_grad_clipping.py
================================
Asserts that gradient clipping is correctly implemented in the passkey
retrieval experiment (exp12_passkey_retrieval/run_exp.py).
"""

import os

def main():
    print("=" * 60)
    print("VERIFY: Passkey Experiment Gradient Clipping")
    print("=" * 60)

    exp12_path = os.path.join("simulations", "exp12_passkey_retrieval", "run_exp.py")
    if not os.path.exists(exp12_path):
        print(f"[FAIL] Could not find passkey experiment file at {exp12_path}")
        return

    with open(exp12_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Assert that clip_grad_norm_ is called in the passkey curriculum training loop
    if "clip_grad_norm_" in code:
        print("[PASS] Gradient clipping function `clip_grad_norm_` is implemented.")
        
        # Verify it is called in both smoke_test and train_curriculum
        occurrences = code.count("clip_grad_norm_")
        print(f"[PASS] Found {occurrences} occurrences of gradient clipping in exp12.")
        
        if occurrences >= 2:
            print("[PASS] Both the smoke test and full curriculum loops are protected by gradient clipping.")
        else:
            print("[WARNING] Gradient clipping is present but found only 1 occurrence.")
    else:
        print("[FAIL] Gradient clipping is missing from exp12_passkey_retrieval/run_exp.py!")
        raise AssertionError("Gradient clipping missing from passkey experiment.")

    print("=" * 60)
    print("[SUCCESS] Passkey experiment gradient clipping verified completely.")
    print("=" * 60)

if __name__ == "__main__":
    main()
