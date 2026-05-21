import torch
import torch.nn.functional as F
import time
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hgdm_omega import OmegaGDM, OmegaConfig

class SyntheticOverfitDataloader:
    def __init__(self, block_size, batch_size):
        self.block_size = block_size
        self.batch_size = batch_size
        # A repeating text string allows the model to actually learn and drop the BPB
        text = "The quick brown fox jumps over the lazy dog. OmegaGDM is learning to route states! "
        # Repeat it enough times to cover the block size
        repeats = (block_size // len(text)) + 2
        full_text = (text * repeats)[:block_size + 1]
        
        self.bytes_data = list(full_text.encode('utf-8'))
        
    def __iter__(self):
        while True:
            # Yield the exact same batch every time to force massive overfitting
            seq = torch.tensor(self.bytes_data, dtype=torch.long)
            batch = seq.unsqueeze(0).repeat(self.batch_size, 1)
            yield batch

def train_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Running on device: {device}")

    # Small configuration for fast testing
    config = OmegaConfig(
        d_byte=256,
        catcher_layers=2,
        renderer_layers=2,
        d_model=256,         # Tiny for testing
        core_layers=4,       # Tiny for testing
        n_heads=8,
        d_k=32,
        d_v=32,
        d_ff=1024,
        decimation_rate=8,
        vocab_size=256,
        use_state_fusion=False
    )
    
    model = OmegaGDM(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[Model] OmegaGDM initialized. Parameters: {params/1e6:.2f}M")

    # Hyperparameters
    max_steps = 1000
    block_size = 512
    batch_size = 8
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    dataloader = SyntheticOverfitDataloader(block_size=block_size, batch_size=batch_size)
    data_stream = iter(dataloader)

    model.train()
    
    print(f"\n{'Step':<5} | {'Loss':<8} | {'BPB':<6} | {'Gate (BU)':<10} | {'Gate (TD)':<10} | {'StepTime':<8}")
    print("-" * 70)
    
    for step in range(max_steps):
        opt.zero_grad(set_to_none=True)
        t_step = time.time()
        
        batch = next(data_stream).to(device)
        x, y = batch[:, :-1], batch[:, 1:]

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        torch.cuda.synchronize()
        step_time = time.time() - t_step
        bpb = loss.item() / math.log(2)
        
        if step % 25 == 0 or step == max_steps - 1:
            # Extract gate values to prove they are learning and moving away from -1.0
            bu_gate_val = model.highway_bu_gate.mean().item()
            td_gate_val = model.highway_td_gate.mean().item()
            
            print(f"{step:04d}  | {loss.item():<8.4f} | {bpb:<6.4f} | {bu_gate_val:<10.4f} | {td_gate_val:<10.4f} | {step_time:.3f}s")
            sys.stdout.flush()

    print("\n[System] Test Complete. Saving checkpoint...")
    checkpoint_path = "trash_100m_1000steps.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'step': max_steps,
        'config': config,
    }, checkpoint_path)
    print(f"[System] Checkpoint saved to: {checkpoint_path}")

    print("\n[System] Generating sample text...")
    model.eval()
    # Use a prompt >= W=8 bytes so the first forward pass completes at least 1 full chunk
    prompt_text = "The quick"
    prompt = torch.tensor([list(prompt_text.encode('utf-8'))], dtype=torch.long, device=device)
    out = model.generate(prompt, max_new_bytes=50)
    print(f"Prompt:  {prompt_text!r}")
    print(f"Output:  {bytes(out[0].tolist()).decode('utf-8', errors='replace')!r}")

    print("\n" + "="*70)
    print("  To run interpretability on this checkpoint:")
    print(f"  python interpret.py --checkpoint {checkpoint_path} --prompt \"The quick brown fox jumps over the lazy dog.\"")
    print("="*70)

if __name__ == "__main__":
    train_test()