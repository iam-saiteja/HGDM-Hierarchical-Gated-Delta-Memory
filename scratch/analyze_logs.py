import json
import math
import sys

def main():
    log_path = "train_100m_logs.jsonl"
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {log_path} not found.")
        return

    steps = []
    losses = []
    train_bpbs = []
    val_bpbs = []
    val_steps = []
    step_times = []

    # Let's assume grad_accum is 4 since train_100m_enwik8.py has default grad_accum = 4
    # We will verify this by comparing training loss/bpb with validation loss/bpb
    grad_accum = 4

    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            step = data.get("step")
            steps.append(step)
            losses.append(data.get("loss"))
            train_bpbs.append(data.get("train_bpb"))
            step_times.append(data.get("step_time"))
            val_bpb = data.get("val_bpb")
            if val_bpb is not None:
                val_bpbs.append(val_bpb)
                val_steps.append(step)
        except Exception as e:
            print(f"Error parsing line: {line.strip()} - {e}")

    if not steps:
        print("No steps found in log file.")
        return

    num_steps = len(steps)
    total_time_s = sum(step_times)
    total_time_h = total_time_s / 3600.0
    avg_step_time = total_time_s / num_steps

    print("==========================================================")
    print("ENWIK8 100M MODEL TRAINING LOGS ANALYSIS")
    print("==========================================================")
    print(f"Total steps logged: {num_steps}")
    print(f"Total training time: {total_time_h:.2f} hours ({total_time_s/60:.1f} minutes)")
    print(f"Average step time: {avg_step_time:.3f} seconds")
    print(f"Estimated parameters: ~126M")
    print("----------------------------------------------------------")

    # Let's print validation steps and values
    print("Validation Progress (every 250 steps):")
    print(f"{'Step':<8} | {'Raw Train Loss':<14} | {'Actual Train Loss':<18} | {'Actual Train BPB':<16} | {'Val BPB':<8}")
    print("-" * 75)
    for s, vb in zip(val_steps, val_bpbs):
        # Find corresponding training loss
        idx = steps.index(s)
        raw_loss = losses[idx]
        actual_loss = raw_loss / grad_accum
        actual_train_bpb = actual_loss / math.log(2)
        print(f"{s:<8d} | {raw_loss:<14.4f} | {actual_loss:<18.4f} | {actual_train_bpb:<16.4f} | {vb:.4f}")
    
    print("----------------------------------------------------------")
    min_val_bpb = min(val_bpbs)
    min_val_step = val_steps[val_bpbs.index(min_val_bpb)]
    print(f"Minimum Validation BPB: {min_val_bpb:.4f} at Step {min_val_step}")
    
    # Check if there is any overfitting
    final_val_bpb = val_bpbs[-1]
    final_train_bpb = (losses[-1] / grad_accum) / math.log(2)
    print(f"Final Train BPB (adjusted): {final_train_bpb:.4f}")
    print(f"Final Val BPB: {final_val_bpb:.4f}")
    print("==========================================================")

if __name__ == "__main__":
    main()
