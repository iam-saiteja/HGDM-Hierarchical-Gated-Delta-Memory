import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision import transforms
from hilbertcurve.hilbertcurve import HilbertCurve
from tqdm import tqdm
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hgdm_omega import OmegaGDM, OmegaConfig

class HilbertVisionDataset(Dataset):
    def __init__(self, hf_dataset, img_size=32):
        self.dataset = hf_dataset
        self.img_size = img_size
        
        # Power of 2 required for standard Hilbert Curve
        self.p = 5 # 2^5 = 32
        self.hilbert = HilbertCurve(self.p, 2)
        
        # Precompute the coordinate mapping
        self.coords = [self.hilbert.point_from_distance(i) for i in range(img_size * img_size)]
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(), # scales to [0, 1]
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['labels'] # 0 for cat, 1 for dog
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        img_tensor = self.transform(image) # (1, 32, 32)
        
        # Scale to [0, 255] byte values
        byte_tensor = (img_tensor * 255).long().squeeze(0) # (32, 32)
        
        # Flatten using Hilbert Curve to preserve spatial locality
        seq = torch.zeros(self.img_size * self.img_size, dtype=torch.long)
        for i, (x, y) in enumerate(self.coords):
            seq[i] = byte_tensor[x, y]
            
        return seq, label

class OmegaVisionClassifier(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # The base model outputs logits of shape (B, T, 256). 
        # We project the final timestep's 256 logits to 2 classes (Cat vs Dog).
        self.classifier = nn.Linear(256, 2)
        
    def forward(self, x):
        # x is (B, T) byte sequence
        
        # Forward pass through the entire OmegaGDM architecture
        logits, _ = self.base_model(byte_seq=x) # (B, T, 256)
        
        # Take the final time-step output
        final_logits = logits[:, -1, :] # (B, 256)
        
        out = self.classifier(final_logits) # (B, 2)
        return out

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading Bingsu/Cat_and_Dog dataset...")
    # It has 'train' and 'test' splits
    ds = load_dataset("Bingsu/Cat_and_Dog", split="train")
    
    # We will use a small subset for quick proof-of-concept
    ds = ds.shuffle(seed=42)
    train_ds = ds.select(range(4000))
    val_ds = ds.select(range(4000, 4500))
    
    train_dataset = HilbertVisionDataset(train_ds, img_size=32)
    val_dataset = HilbertVisionDataset(val_ds, img_size=32)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize base OmegaGDM config
    cfg = OmegaConfig(
        d_byte=128, catcher_layers=1, renderer_layers=1, # smaller byte layers
        d_model=384, core_layers=4, n_heads=6, # smaller core for fast training
        d_k=64, d_v=64, d_ff=1536,
        decimation_rate=4, max_position_embeddings=1024, # 32x32 = 1024
        vocab_size=256, use_state_fusion=False
    )
    
    base_model = OmegaGDM(cfg, force_sequential=False)
    model = OmegaVisionClassifier(base_model).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("\nStarting Training (Continuous-Time Vision via Hilbert Scan)...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for seqs, labels in pbar:
            seqs, labels = seqs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.2f}%"})
            
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for seqs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                seqs, labels = seqs.to(device), labels.to(device)
                logits = model(seqs)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        print(f"Epoch {epoch+1} Summary | Train Acc: {100.*correct/total:.2f}% | Val Acc: {100.*val_correct/val_total:.2f}% | Val Loss: {val_loss/len(val_loader):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()
    
    train(args)
