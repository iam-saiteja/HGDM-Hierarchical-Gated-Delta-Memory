import torch
import torch.nn.functional as F
import os
import sys
import random
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hgdm_omega import OmegaGDM, OmegaConfig

# Custom Identity Data to ensure the model knows who it is!
IDENTITY_DATA = [
    {"instruction": "Who are you?", "input": "", "output": "I am Omega Edge, a tiny continuous-time AI model running natively on your smartwatch."},
    {"instruction": "What can you do?", "input": "", "output": "I can chat with you, answer basic questions, and run entirely locally without an internet connection!"},
    {"instruction": "What is your architecture?", "input": "", "output": "I am a 35-million parameter continuous-time OmegaGDM model. I don't use Transformers or positional embeddings!"},
    {"instruction": "Tell me about yourself.", "input": "", "output": "I am Omega Edge. I was built to run efficiently on edge hardware like smartwatches by processing continuous time."}
]

class ChatDataset(IterableDataset):
    def __init__(self, hf_dataset, seq_len=512, dream_prob=0.5):
        self.dataset = hf_dataset
        self.seq_len = seq_len
        self.dream_prob = dream_prob
        self.identity_data = IDENTITY_DATA * 500 # Weight the identity data heavily

    def __iter__(self):
        buffer_x = []
        buffer_y = []
        
        # Interleave identity data with Alpaca data
        all_data = list(self.dataset) + self.identity_data
        random.shuffle(all_data)
        
        for item in all_data:
            # Format the prompt
            user_text = item['instruction']
            if item.get('input', '') != '':
                user_text += f"\n{item['input']}"
                
            prompt = f"User: {user_text}\nOmega: "
            answer = f"{item['output']}\n\n"
            
            prompt_bytes = list(prompt.encode('utf-8', errors='ignore'))
            answer_bytes = list(answer.encode('utf-8', errors='ignore'))
            
            # --- DREAMING WHILE TRAINING ---
            # We randomly inject 3 to 10 'Null' bytes before the answer.
            # The model sees them (x) and updates its ODE state, 
            # but the loss function ignores them (y = -100).
            dream_length = random.randint(3, 10) if random.random() < self.dream_prob else 0
            dream_bytes = [0x00] * dream_length
            
            # Construct Input (x)
            x_seq = prompt_bytes + dream_bytes + answer_bytes
            
            # Construct Target (y)
            # -100 tells PyTorch CrossEntropyLoss to ignore this prediction
            y_seq = [-100] * len(prompt_bytes) + [-100] * len(dream_bytes) + answer_bytes
            
            # Offset by 1 for next-byte prediction
            x_seq = x_seq[:-1]
            y_seq = y_seq[1:]
            
            buffer_x.extend(x_seq)
            buffer_y.extend(y_seq)
            
            while len(buffer_x) >= self.seq_len:
                x = buffer_x[:self.seq_len]
                y = buffer_y[:self.seq_len]
                buffer_x = buffer_x[self.seq_len:]
                buffer_y = buffer_y[self.seq_len:]
                yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def finetune_edge():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 35M Parameter Configuration
    config = OmegaConfig(
        d_byte=256, catcher_layers=2, renderer_layers=2, 
        d_model=256, core_layers=6, n_heads=8, 
        d_k=32, d_v=32, d_ff=1024, decimation_rate=8, 
        max_position_embeddings=512, vocab_size=256, use_state_fusion=False
    )
    
    model = OmegaGDM(config, force_sequential=False).to(device)
    
    # Load the Pre-Trained 35M Chinchilla Model!
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "scaling", "omega_chinchilla_35m.pt")
    if os.path.exists(ckpt_path):
        print(f"[*] Loading Pre-Trained Chinchilla Base Model: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    else:
        print(f"[!] Critical Error: Pre-Trained Model not found at {ckpt_path}. We need this to ensure it understands English!")
        return

    # Load high-quality tiny dataset
    print("[*] Loading Alpaca Cleaned (51k conversational turns)...")
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    
    # We use a very small batch size and learning rate to carefully instruction-tune 
    # without wiping out its pre-trained knowledge.
    seq_len = 512
    batch_size = 16
    lr = 2e-4
    steps = 5000  # A robust fine-tuning run
    
    train_loader = DataLoader(ChatDataset(ds, seq_len=seq_len, dream_prob=0.8), batch_size=batch_size, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    model.train()
    train_iter = iter(train_loader)
    
    print("\n==================================================")
    print("Starting Omega Edge Instruction-Tuning (with Dreaming)")
    print("==================================================")
    
    pbar = tqdm(range(steps), desc="Fine-Tuning Omega Edge")
    for step in pbar:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
            
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(x)
            # ignore_index=-100 ensures the model isn't penalized for the prompt or the "dreaming" bytes
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), ignore_index=-100)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    out_path = os.path.join(os.path.dirname(__file__), "omega_edge_v1.pt")
    torch.save(model.state_dict(), out_path)
    print(f"\n[*] Training Complete! Edge Model saved to {out_path}")

if __name__ == "__main__":
    finetune_edge()
