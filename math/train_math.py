import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import time
import os
import math
import sys
import subprocess
from hgdm_ultimate import HGDMUltimate, HGDMConfig

# =============================================================================
# 1. MATH MISSION CONFIGURATION (V3.0 STEEL)
# =============================================================================
BATCH_SIZE = 1 
SEQ_LEN = 2048      
ACCUM_STEPS = 4     
INITIAL_LR = 5.0e-6   # SURGICAL STABILITY
SAVE_EVERY = 500    
CHECKPOINT_START = "math_start.pt"
TARGET_TOKENS = 5_000_000_000 

# Thermal Redline (88C Target)
TARGET_TEMP = 88.0
SOFT_LIMIT = 90.0
HARD_LIMIT = 92.0

# ... (Dataloader and Controller remain same) ...

# =============================================================================
# 2. DATASET: OPEN-WEB-MATH (STREAMING)
# =============================================================================
class MathChunkDataset(IterableDataset):
    def __init__(self, hf_ds, seq_len=2048, buffer_size=0x200000):  # 2 MB limit
        self.hf_ds = hf_ds
        self.seq_len = seq_len
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer = bytearray()
        for sample in self.hf_ds:
            text = sample['text']
            # Robust byte encoding
            data = text.encode('utf-8', errors='ignore')
            buffer.extend(data)
            
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                x = torch.tensor(list(chunk[:-1]), dtype=torch.long)
                y = torch.tensor(list(chunk[1:]), dtype=torch.long)
                yield x, y
                buffer = buffer[self.seq_len:]   # advance by seq_len
                
            if len(buffer) > self.buffer_size:
                buffer = buffer[-self.seq_len:]  # Guard against unbounded growth

def get_math_dataloader(seq_len=2048):
    print(">>> Connecting to OpenWebMath Stream...")
    ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    math_ds = MathChunkDataset(ds, seq_len=seq_len)
    return DataLoader(math_ds, batch_size=BATCH_SIZE)

# =============================================================================
# 3. NITRO-THERMAL PID (REDLINE CALIBRATION)
# =============================================================================
class NitroThermalController:
    def __init__(self, target_temp=88.0, Kp=0.3, Ki=0.02, Kd=0.25):
        self.target_temp = target_temp
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.prev_error = 0
        self.integral = 0
        self.smoothed_temp = None
        self.alpha = 0.3
        self.sleep_time = 0.0
        self.integral_clamp = 15

    def reset(self, temp):
        self.smoothed_temp = temp
        self.integral = 0
        self.prev_error = 0

    def step(self, temp):
        if self.smoothed_temp is None: self.smoothed_temp = temp
        else: self.smoothed_temp = (1 - self.alpha) * self.smoothed_temp + self.alpha * temp
        
        # Redline Safety
        if temp >= HARD_LIMIT: 
            self.reset(temp)
            time.sleep(60.0)
            return 60.0, "EMERGENCY"
        if temp >= SOFT_LIMIT: 
            self.reset(temp)
            time.sleep(15.0)
            return 15.0, "SABBATH"
        
        error = self.smoothed_temp - self.target_temp
        self.integral = max(-self.integral_clamp, min(self.integral_clamp, self.integral + error))
        derivative = error - self.prev_error
        
        adjustment = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.sleep_time = max(0.0, min(10.0, self.sleep_time + adjustment))
        self.prev_error = error
        
        if self.sleep_time > 0.1:
            time.sleep(self.sleep_time)
            return self.sleep_time, "PID_PAUSE"
        return 0.0, "OK"

# =============================================================================
# 4. UTILITIES
# =============================================================================
class TitanLogger:
    def __init__(self, log_file="math_training_logs.md"):
        self.log_file = log_file
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("# 📐 TITAN-MATH-1B Training Log\n\n")
                f.write("| Step | Tokens | BPB | Time | GPU | VRAM | Sabbath | Status |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

    def log_step(self, step, tokens, bpb, dt, temp, vram, cool_min, status):
        with open(self.log_file, "a") as f:
            f.write(f"| {step} | {tokens/1e6:.1f}M | {bpb:.4f} | {dt:.2f}s | {temp}C | {vram}MB | {cool_min:.1f}m | {status} |\n")

def get_gpu_stats():
    try:
        cmd = "nvidia-smi --query-gpu=temperature.gpu,memory.used,utilization.gpu --format=csv,noheader,nounits"
        out = subprocess.check_output(cmd, shell=True).decode().strip().split(',')
        return int(out[0]), int(out[1]), int(out[2])
    except: return 0, 0, 0

# =============================================================================
# 5. MISSION START: MATH PRE-TRAINING
# =============================================================================
def launch_math_mission():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = TitanLogger()
    thermal_controller = NitroThermalController()
    os.makedirs("math_checkpoints", exist_ok=True)
    
    # 1. Load Architecture
    config = HGDMConfig(d_model=1792, n_layers=20, n_heads=28, d_ff=7168)
    model = HGDMUltimate(config).to(device)
    
    # 2. Dynamic Resume (Latest Checkpoint -> Start Checkpoint)
    start_step = 1
    total_tokens = 0
    resume_path = "math_latest.pt" if os.path.exists("math_latest.pt") else CHECKPOINT_START
    
    # 3. Optimization (8-bit AdamW)
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=INITIAL_LR, weight_decay=0.1)
        print(">>> 8-bit AdamW Initialized.")
    except ImportError:
        optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.1)

    if os.path.exists(resume_path):
        print(f">>> Resuming Math Mission from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(">>> Optimizer state restored (Full Momentum).")
        start_step = checkpoint.get("step", 1) + 1
        total_tokens = checkpoint.get("total_tokens", 0)
        del checkpoint
        torch.cuda.empty_cache()
    
    dataloader = get_math_dataloader(seq_len=SEQ_LEN)
    
    # 4. DATA STREAM FAST-FORWARD
    if total_tokens > 0:
        # JUMP PROTOCOL: Add 2000 batches to skip the lethal minefield at Step 7490
        batches_to_skip = ((start_step - 1) * ACCUM_STEPS) + 2000 
        print(f">>> JUMP PROTOCOL ACTIVE: Skipping {batches_to_skip} math batches...")
        data_iter = iter(dataloader)
        for _ in range(batches_to_skip):
            try: next(data_iter)
            except StopIteration: break
        print(">>> Jump Successful. Landed in clean data sector.")
    else:
        data_iter = iter(dataloader)
    scaler = torch.amp.GradScaler('cuda') # Initialize AMP
    
    print("\n" + "="*60)
    print("🚀 TITAN-MATH-1B: PHASE 1 START")
    print("="*60)
    
    mission_start = time.time()
    total_cooldown_s = 0
    torch.cuda.empty_cache()
    
    try:
        for step in range(start_step, 1000000):
            step_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            
            # Gradient Accumulation
            accum_loss = 0
            for _ in range(ACCUM_STEPS):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    dataloader = get_math_dataloader(seq_len=SEQ_LEN)
                    data_iter = iter(dataloader)
                    x, y = next(data_iter)
                
                x, y = x.to(device), y.to(device)
                
                # Mixed Precision Forward (BFLOAT16 FOR STABILITY)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = model(x)
                    
                    # LOGIT SANITY CHECK
                    if torch.max(torch.abs(logits)) > 5000.0:
                        print("\n🚨 EXTREME WEIGHT TURBULENCE DETECTED. ABORTING.")
                        sys.exit(1)
                        
                    loss = nn.CrossEntropyLoss()(logits.view(-1, 256), y.view(-1))
                
                scaler.scale(loss / ACCUM_STEPS).backward()
                accum_loss += loss.item()
                total_tokens += x.numel()
            
            # Scaler Step
            scaler.unscale_(optimizer)
            
            # NaN CIRCUIT BREAKER
            if math.isnan(accum_loss / ACCUM_STEPS):
                print("\n" + "!"*60)
                print("🚨 CRITICAL FAILURE: NaN DETECTED. MISSION ABORTED.")
                print("!"*60)
                sys.exit(1)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Tightened clipping
            scaler.step(optimizer)
            scaler.update()
            
            if step % 25 == 0:
                torch.cuda.empty_cache() # Regular flush
            
            # Stats & Thermal
            temp, vram, util = get_gpu_stats()
            bpb = (accum_loss / ACCUM_STEPS) / math.log(2)
            pause_duration, status = thermal_controller.step(temp)
            total_cooldown_s += pause_duration
            
            dt = time.time() - step_start
            
            if step % 10 == 0:
                elapsed = time.time() - mission_start
                cool_min = total_cooldown_s / 60
                print(f"Step {step:6d} | Tokens: {total_tokens/1e6:6.1f}M | BPB: {bpb:.4f} | Time: {dt:.2f}s | GPU: {temp}C | {vram}MB | Sabbath: {cool_min:.1f}m | Status: {status}")
                logger.log_step(step, total_tokens, bpb, dt, temp, vram, cool_min, status)

            if step % SAVE_EVERY == 0:
                ckpt_path = f"math_checkpoints/math_step_{step}.pt"
                state = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "total_tokens": total_tokens
                }
                torch.save(state, ckpt_path)
                torch.save(state, "math_latest.pt")
                print(f"--> Saved Math Checkpoint at Step {step}")

    except KeyboardInterrupt:
        print("\n\n>>> INTERRUPT DETECTED. SAVING EMERGENCY MATH CHECKPOINT...")
        state = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "total_tokens": total_tokens
        }
        torch.save(state, "math_latest.pt")
        print(">>> Emergency Checkpoint Saved. Exiting.")

if __name__ == "__main__":
    launch_math_mission()
