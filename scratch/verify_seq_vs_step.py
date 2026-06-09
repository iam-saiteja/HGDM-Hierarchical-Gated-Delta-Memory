import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from hgdm_omega import OmegaConfig, OmegaGDM

def check_state_diff(s1, s2, label=""):
    if s1 is None and s2 is None:
        return
    if s1 is None or s2 is None:
        print(f"  [State Diff] {label}: one is None")
        return
    
    if isinstance(s1, tuple):
        for idx, (t1, t2) in enumerate(zip(s1, s2)):
            if t1 is not None and t2 is not None:
                diff = (t1 - t2).abs().max().item()
                if diff > 1e-4:
                    print(f"  [State Diff] {label} tuple element {idx} diff: {diff}")
            elif t1 is not None or t2 is not None:
                print(f"  [State Diff] {label} tuple element {idx}: one is None")
    elif isinstance(s1, dict):
        for k in s1.keys():
            t1, t2 = s1[k], s2[k]
            if t1 is None and t2 is None:
                continue
            if t1 is None or t2 is None:
                print(f"  [State Diff] {label} dict key {k}: one is None")
                continue
            if torch.is_tensor(t1):
                diff = (t1 - t2).float().abs().max().item()
                if diff > 1e-4:
                    print(f"  [State Diff] {label} dict key {k} diff: {diff}")
    elif torch.is_tensor(s1):
        diff = (s1 - s2).abs().max().item()
        if diff > 1e-4:
            print(f"  [State Diff] {label} tensor diff: {diff}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Simple model config
    config = OmegaConfig(
        catcher_layers=1, renderer_layers=1, core_layers=2,
        d_byte=64, d_model=256, n_heads=4, d_k=32, d_v=32, d_ff=512,
        decimation_rate=8, max_position_embeddings=256
    )
    model = OmegaGDM(config).to(device)
    model.eval()
    
    # Input sequence (e.g. 16 steps)
    B, T = 1, 16
    x = torch.randint(0, 256, (B, T), device=device)
    
    print("\n--- Parallel Forward ---")
    with torch.no_grad():
        logits_seq, states_seq = model(x, return_states=True)
    
    print("\n--- Step-by-Step Forward ---")
    states_step = None
    logits_step_list = []
    
    with torch.no_grad():
        for t in range(T):
            token = x[:, t:t+1]
            logits_t, states_step = model(token, states=states_step, offset=t, return_states=True)
            logits_step_list.append(logits_t)
            
    logits_step = torch.cat(logits_step_list, dim=1)
    
    # Compare logits
    diff_logits = (logits_seq - logits_step).abs().max().item()
    print(f"\nMax Logits Difference: {diff_logits:.6f}")
    if diff_logits > 1e-4:
        print("❌ LOGITS DISCREPANCY DETECTED!")
        # Let's inspect step-by-step diffs
        for t in range(T):
            step_diff = (logits_seq[:, t] - logits_step[:, t]).abs().max().item()
            print(f"  Step {t}: {step_diff:.6f}")
    else:
        print("✅ Logits match!")
        
    # Compare states
    print("\nComparing states...")
    # Catcher layers
    for i in range(config.catcher_layers):
        check_state_diff(states_seq[0][i], states_step[0][i], f"Catcher layer {i}")
    # Core layers
    for i in range(config.core_layers):
        check_state_diff(states_seq[1][i], states_step[1][i], f"Core layer {i}")
    # Renderer layers
    for i in range(config.renderer_layers):
        check_state_diff(states_seq[2][i], states_step[2][i], f"Renderer layer {i}")
    # Decimator states
    check_state_diff(states_seq[3], states_step[3], "Decimator state")
    # Buffer dictionary
    check_state_diff(states_seq[4], states_step[4], "Buffer state")

if __name__ == "__main__":
    main()
