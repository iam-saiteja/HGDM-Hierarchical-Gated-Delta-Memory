import torch
import torch.nn as nn
import os
import urllib.request
import zipfile
import subprocess

def get_gpu_memory_usage():
    """Queries nvidia-smi for the exact 'Memory-Usage' value shown in the terminal."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        return int(result.strip())
    except Exception:
        # Fallback to PyTorch reserved memory if nvidia-smi fails
        return int(torch.cuda.memory_reserved() / (1024**2))

def get_enwik8_data():
    """Downloads and returns Enwik8 data as a uint8 tensor."""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "enwik8.zip")
    data_path = os.path.join(data_dir, "enwik8")
    
    if not os.path.exists(data_path):
        if not os.path.exists(zip_path):
            print("Downloading enwik8 (100MB)...")
            url = "http://mattmahoney.net/dc/enwik8.zip"
            urllib.request.urlretrieve(url, zip_path)
        print("Extracting enwik8...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
    with open(data_path, 'rb') as f:
        data = f.read()
    
    n = len(data)
    train_data = torch.frombuffer(data[:int(n * 0.9)], dtype=torch.uint8).long()
    val_data = torch.frombuffer(data[int(n * 0.9):], dtype=torch.uint8).long()
    return train_data, val_data

@torch.no_grad()
def evaluate_model(model, val_data, seq_len=2048, batches=20):
    device = next(model.parameters()).device
    model.eval()
    
    total_loss = 0.0
    for _ in range(batches):
        ix = torch.randint(len(val_data) - seq_len - 1, (1,))
        x = torch.stack([val_data[i:i+seq_len] for i in ix]).to(device)
        y = torch.stack([val_data[i+1:i+seq_len+1] for i in ix]).to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
            if isinstance(out, tuple): out = out[0]
            loss = torch.nn.functional.cross_entropy(out.view(-1, 256), y.view(-1))
            total_loss += loss.item()
            
    model.train()
    return total_loss / batches

from hgdm_ultimate import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, use_flash=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_flash = use_flash
        self.norm1 = nn.LayerNorm(d_model)
        if use_flash:
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x, mask=None):
        nx = self.norm1(x)
        if self.use_flash:
            attn_out, _ = self.attn(nx, nx, nx, attn_mask=mask, is_causal=True, need_weights=False)
        else:
            # Manual O(N^2) Attention to show memory explosion without Flash/Tricks
            B, T, C = nx.shape
            head_dim = C // self.n_heads
            qkv = nx.view(B, T, self.n_heads, head_dim).transpose(1, 2)
            q, k, v = qkv, qkv, qkv # Simplification for scaling test
            dots = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
            # Causal Mask
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            dots = dots.masked_fill(mask, float('-inf'))
            attn = torch.softmax(dots, dim=-1)
            attn_out = (attn @ v).transpose(1, 2).reshape(B, T, C)
            
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class BaselineTransformer(nn.Module):
    """
    A 120M parameter standard Transformer to perfectly match HGDM-120M.
    Uses learned positional embeddings, standard self-attention, and SwiGLU.
    """
    def __init__(self, d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=256, max_seq_len=16384, use_flash=True):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, use_flash=use_flash) for _ in range(n_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)
        self.fc_out.weight = self.embedding.weight
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        
        x = self.embedding(x) + self.pos_embedding(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        
        for layer in self.layers:
            x = layer(x, mask=mask)
            
        x = self.norm_f(x)
        return self.fc_out(x)
