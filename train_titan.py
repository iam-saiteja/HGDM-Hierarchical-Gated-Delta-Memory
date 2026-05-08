import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
import subprocess
import json
from hgdm_ultimate import HGDMUltimate, HGDMConfig
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

# =============================================================================
# 1. TITAN-1B CONFIGURATION
# =============================================================================
titan_config = HGDMConfig(
    d_model=1792,
    n_layers=20,
    n_heads=28,
    d_k=64,
    d_v=64,
    d_ff=7168,
    vocab_size=256
)

# =============================================================================
# 2. STREAMING DATALOADER
# =============================================================================
class ByteChunkDataset(IterableDataset):
    def __init__(self, dataset_stream, seq_len=2048, buffer_size=0x100000):
        self.dataset_stream = dataset_stream
        self.seq_len = seq_len
        self.buffer = bytearray()
        self.buffer_size = buffer_size

    def __iter__(self):
        for example in self.dataset_stream:
            text = example["text"]
            data = text.encode("utf-8", errors="ignore")
            self.buffer.extend(data)
            while len(self.buffer) >= self.seq_len + 1:
                chunk = self.buffer[:self.seq_len + 1]
                self.buffer = self.buffer[self.seq_len + 1:]
                input_bytes = torch.tensor(list(chunk[:-1]), dtype=torch.long)
                target_bytes = torch.tensor(list(chunk[1:]), dtype=torch.long)
                yield input_bytes, target_bytes
            if len(self.buffer) > self.buffer_size:
                self.buffer = self.buffer[-self.seq_len:]

def get_streaming_dataloader(batch_size=1, seq_len=2048):
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    byte_dataset = ByteChunkDataset(ds, seq_len=seq_len)
    return DataLoader(byte_dataset, batch_size=batch_size)

# =============================================================================
# 3. LOGGING & REPORTING SUITE
# =============================================================================
class TitanLogger:
    def __init__(self, filename="titan_training_logs.md"):
        self.filename = filename
        if not os.path.exists(self.filename):
            with open(self.filename, "w") as f:
                f.write("# 🪐 TITAN-1B Training Log\n\n")
                f.write("| Step | BPB | Time | GPU | VRAM | LR | Tokens | Status |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

    def log_step(self, step, bpb, dt, temp, vram, lr, tokens, status="OK"):
        with open(self.filename, "a") as f:
            tokens_m = tokens / 1e6
            f.write(f"| {step} | {bpb:.4f} | {dt:.2f}s | {temp}C | {vram}MB | {lr:.2e} | {tokens_m:.1f}M | {status} |\n")

    def log_sample(self, step, prompt, response, tag="Sample"):
        with open(self.filename, "a") as f:
            f.write(f"\n### Step {step} {tag}\n")
            f.write(f"**Prompt:** `{prompt}`\n\n")
            f.write(f"**Response:**\n```text\n{response}\n```\n\n")

# =============================================================================
# 4. UTILITIES
# =============================================================================
def get_gpu_stats():
    try:
        cmd = "nvidia-smi --query-gpu=temperature.gpu,memory.used,utilization.gpu --format=csv,noheader,nounits"
        out = subprocess.check_output(cmd, shell=True).decode().strip().split(',')
        return int(out[0]), int(out[1]), int(out[2])
    except: return 0, 0, 0
    
class NitroThermalController:
    def __init__(self, target_temp=83.0, Kp=0.25, Ki=0.015, Kd=0.20):
        self.target_temp = target_temp
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.prev_error = 0
        self.integral = 0
        self.smoothed_temp = None
        self.alpha = 0.3  # Faster EMA for quicker reaction
        self.sleep_time = 0.0
        self.integral_clamp = 15

    def reset(self, temp):
        self.smoothed_temp = temp
        self.integral = 0
        self.prev_error = 0
        self.sleep_time = 2.0 # Start with a cooling cushion

    def step(self, temp):
        # 1. EMA Smoothing
        if self.smoothed_temp is None: self.smoothed_temp = temp
        else: self.smoothed_temp = (1 - self.alpha) * self.smoothed_temp + self.alpha * temp
        
        # 2. Safety Overrides (with state resets)
        if temp >= 91: 
            self.reset(temp)
            time.sleep(60.0)
            return 60.0, "EMERGENCY"
        if temp >= 89: 
            self.reset(temp)
            time.sleep(15.0)
            return 15.0, "HARD_PAUSE"
        
        # 3. PID Logic
        error = self.smoothed_temp - self.target_temp
        self.integral = max(-self.integral_clamp, min(self.integral_clamp, self.integral + error))
        derivative = error - self.prev_error
        
        # Calculate Adjustment
        adjustment = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.sleep_time = max(0.0, min(8.0, self.sleep_time + adjustment))
        self.prev_error = error
        
        # 4. Apply PID Throttling
        if self.sleep_time > 0.1:
            time.sleep(self.sleep_time)
            return self.sleep_time, "PID_PAUSE"
        return 0.0, "OK"

@torch.no_grad()
def generate_sample(model, device, prompt="Wikipedia is ", length=100):
    model.eval()
    input_ids = torch.tensor([list(prompt.encode('utf-8'))], dtype=torch.long).to(device)
    generated_ids = model.generate(input_ids, max_new_bytes=length)
    model.train()
    return bytes(generated_ids[0].tolist()).decode('utf-8', errors='ignore')

# =============================================================================
# 5. MAIN MISSION: THE TITAN LAUNCH
# =============================================================================
def launch_titan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = TitanLogger()
    thermal_controller = NitroThermalController() # Initialize PID
    os.makedirs("titan_checkpoints", exist_ok=True)
    
    print(f"Initializing TITAN-1B on {device}...")
    torch.cuda.empty_cache()
    model = HGDMUltimate(titan_config).to(device)
    
    start_step = 1
    total_tokens = 0
    ACCUM_STEPS = 4
    if os.path.exists("titan_latest.pt"):
        # Load to CPU first to avoid VRAM spike
        checkpoint = torch.load("titan_latest.pt", map_location='cpu')
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            start_step = checkpoint.get("step", 1) + 1
            total_tokens = checkpoint.get("total_tokens", 0)
        else:
            model.load_state_dict(checkpoint)
        
        print(f">>> Resuming Mission from Step {start_step} ({total_tokens/1e6:.1f}M tokens)...")
        del checkpoint # Delete CPU copy immediately
        torch.cuda.empty_cache()
    
    # Learning Rate Calibration (4.5e-5 to match previous run exit)
    initial_lr = 4.5e-5
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=initial_lr)
        print(f"Using 8-bit AdamW (Initial LR: {initial_lr:.2e})")
    except:
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
        print(f"WARNING: Standard AdamW (Initial LR: {initial_lr:.2e})")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20000, eta_min=2e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    # Data Stream Fast-Forward
    print(f">>> Resuming data stream from {total_tokens/1e6:.1f}M tokens...")
    train_loader = get_streaming_dataloader()
    train_iter = iter(train_loader)
    
    # Fast-forwarding the iterator
    if total_tokens > 0:
        steps_to_skip = (start_step - 1) * ACCUM_STEPS
        print(f">>> Fast-forwarding {steps_to_skip} batches...")
        for _ in range(steps_to_skip):
            try: next(train_iter)
            except StopIteration: break
    
    mission_start = time.time()
    MAX_DURATION = 12 * 3600 
    total_cooldown_s = 0
    
    print("\n" + "="*60 + "\nTITAN-1B MISSION RESUME" if start_step > 1 else "\nTITAN-1B MISSION START")
    print("="*60)
    
    try:
        for step in range(start_step, 1000001):
            elapsed = time.time() - mission_start
            if elapsed > MAX_DURATION:
                print(f"\nMISSION COMPLETE.")
                break
                
            step_start = time.time()
            accum_loss = 0
            optimizer.zero_grad()
            
            for _ in range(ACCUM_STEPS):
                try: x, y = next(train_iter)
                except: train_iter = iter(train_loader); x, y = next(train_iter)
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
                    loss = loss / ACCUM_STEPS
                scaler.scale(loss).backward()
                accum_loss += loss.item() * ACCUM_STEPS
                total_tokens += x.numel()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            curr_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            
            temp, vram, util = get_gpu_stats()
            bpb = accum_loss / math.log(2)
            
            # --- NITRO-THERMAL PID CONTROL ---
            pause_duration, status = thermal_controller.step(temp)
            total_cooldown_s += pause_duration
            
            dt = time.time() - step_start
            
            # Consolidated Console Output
            if step <= 10 or step % 25 == 0:
                eta_h = (MAX_DURATION - elapsed) / 3600
                cool_min = total_cooldown_s / 60
                print(f"Step {step:5d} | BPB: {bpb:.4f} | Time: {dt:.2f}s | GPU: {temp}C | {vram}MB | {util}% | LR: {curr_lr:.2e} | Sabbath: {cool_min:.1f}m | ETA: {eta_h:.1f}h")
                
            if step % 50 == 0:
                logger.log_step(step, bpb, dt, temp, vram, curr_lr, total_tokens, status)
                
            if step % 500 == 0:
                checkpoint_data = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "total_tokens": total_tokens
                }
                torch.save(checkpoint_data, f"titan_checkpoints/step_{step}.pt")
                torch.save(checkpoint_data, "titan_latest.pt")
                print(f"--> Saved Checkpoint at Step {step}")

    except KeyboardInterrupt:
        print("\n\n>>> INTERRUPT DETECTED. SAVING EMERGENCY CHECKPOINT...")
        checkpoint_data = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "total_tokens": total_tokens
        }
        torch.save(checkpoint_data, "titan_latest.pt")
        print(">>> Emergency Checkpoint Saved. Titan is safe. Exiting.")

    torch.save(model.state_dict(), "titan_final.pt")

if __name__ == "__main__":
    launch_titan()
