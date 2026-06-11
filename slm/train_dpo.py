import torch
import torch.nn.functional as F
import time
import os
import sys
import argparse
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hgdm_omega import OmegaGDM, OmegaConfig
from slm.data_dpo import get_dpo_dataloader

def compute_logprobs(model, x, y):
    """
    Computes the sum of log-probabilities for the true target tokens.
    Tokens with target = -100 are ignored (user prompts).
    """
    logits, _ = model(x)
    
    valid_mask = (y != -100)
    y_safe = torch.where(valid_mask, y, torch.zeros_like(y))
    
    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logprobs = log_probs.gather(dim=-1, index=y_safe.unsqueeze(-1)).squeeze(-1)
    
    return (per_token_logprobs * valid_mask).sum(dim=1)

def train_dpo():
    parser = argparse.ArgumentParser(description="Train DPO OmegaGDM SLM")
    parser.add_argument("--sft-ckpt", type=str, default="omega_slm_sft_latest.pt", help="Path to the SFT checkpoint")
    parser.add_argument("--steps", type=int, default=5000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=type(1e-5), default=1e-5, help="Peak learning rate for DPO (usually much smaller than SFT)")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO KL penalty coefficient")
    parser.add_argument("--block-size", type=int, default=2048, help="Sequence block length")
    parser.add_argument("--ckpt", type=str, default="omega_slm_dpo_latest.pt", help="Checkpoint save path")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Device selected: {device}")

    omega_cfg = OmegaConfig(
        d_byte=256, catcher_layers=2, renderer_layers=2,
        d_model=768, core_layers=12, n_heads=12,
        d_k=64, d_v=64, d_ff=3072,
        decimation_rate=8, max_position_embeddings=args.block_size,
        vocab_size=256, use_state_fusion=False
    )
    
    print("[Model] Initializing Reference Model and Policy Model...")
    model = OmegaGDM(omega_cfg, force_sequential=False).to(device)
    
    if not os.path.exists(args.sft_ckpt):
        print(f"[CRITICAL ERROR] SFT Checkpoint {args.sft_ckpt} not found! DPO requires an SFT model.")
        sys.exit(1)
        
    print(f"[System] Loading SFT checkpoint: {args.sft_ckpt}")
    ckpt = torch.load(args.sft_ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    
    # Create the frozen reference model
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    dataloader = get_dpo_dataloader(batch_size=args.batch_size, block_size=args.block_size)
    data_stream = iter(dataloader)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    print("=========================================")
    print("STARTING DIRECT PREFERENCE OPTIMIZATION (DPO)")
    print("=========================================")
    model.train()
    
    t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_margin = 0.0
        
        for _ in range(args.grad_accum):
            try:
                xc, yc, xr, yr = next(data_stream)
            except StopIteration:
                dataloader = get_dpo_dataloader(batch_size=args.batch_size, block_size=args.block_size)
                data_stream = iter(dataloader)
                xc, yc, xr, yr = next(data_stream)
                
            xc, yc, xr, yr = xc.to(device), yc.to(device), xr.to(device), yr.to(device)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.bfloat16):
                # Compute Reference LogProbs (no gradients needed)
                with torch.no_grad():
                    ref_logprobs_chosen = compute_logprobs(ref_model, xc, yc)
                    ref_logprobs_rejected = compute_logprobs(ref_model, xr, yr)
                    
                # Compute Policy LogProbs
                pi_logprobs_chosen = compute_logprobs(model, xc, yc)
                pi_logprobs_rejected = compute_logprobs(model, xr, yr)
                
                # DPO Loss calculation
                pi_logratios = pi_logprobs_chosen - pi_logprobs_rejected
                ref_logratios = ref_logprobs_chosen - ref_logprobs_rejected
                
                logits = pi_logratios - ref_logratios
                loss = -F.logsigmoid(args.beta * logits).mean()
                
                loss = loss / args.grad_accum
                
            scaler.scale(loss).backward()
            accum_loss += loss.item() * args.grad_accum
            accum_margin += logits.mean().item() / args.grad_accum
            
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        
        if step % 10 == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"DPO Step {step:4d} | Loss: {accum_loss:.4f} | Reward Margin: {accum_margin:.4f} | Time/10-steps: {dt:.2f}s")
            
        if step > 0 and step % 250 == 0:
            torch.save({
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'step': step
            }, args.ckpt)

if __name__ == "__main__":
    train_dpo()
