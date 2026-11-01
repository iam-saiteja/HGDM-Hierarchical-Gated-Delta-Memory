import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenized.config import TokenizedConfig
from tokenized.model_hgdm import TokenizedHGDM
from transformers import AutoTokenizer

def fuse_model_projections(model: TokenizedHGDM):
    """
    Fuses the 6 individual linear projections in each TokenizedMHGD layer
    (W_q, W_k, W_v, W_beta, W_delta, W_out) into a single GEMM projection W_in.
    """
    for layer_idx, layer in enumerate(model.layers):
        mixer = layer.mixer
        H = mixer.H
        d_k = mixer.d_k
        d_v = mixer.d_v
        d_model = mixer.d_model
        
        d_q = H * d_k
        d_k_proj = H * d_k
        d_v_proj = H * d_v
        d_beta = H
        d_delta = H
        d_out = H * d_v
        
        total_dim = d_q + d_k_proj + d_v_proj + d_beta + d_delta + d_out
        dev = mixer.W_q.weight.device
        dtype = mixer.W_q.weight.dtype
        
        W_in = nn.Linear(d_model, total_dim, bias=True, device=dev, dtype=dtype)
        
        with torch.no_grad():
            w_delta_w = mixer.W_delta.weight if mixer.config.use_variable_delta_t else mixer.W_alpha.weight
            w_delta_b = mixer.W_delta.bias if mixer.config.use_variable_delta_t else mixer.W_alpha.bias
            
            W_in.weight.copy_(torch.cat([
                mixer.W_q.weight,
                mixer.W_k.weight,
                mixer.W_v.weight,
                mixer.W_beta.weight,
                w_delta_w,
                mixer.W_out.weight
            ], dim=0))
            
            W_in.bias.copy_(torch.cat([
                torch.zeros(d_q, device=dev, dtype=dtype),
                torch.zeros(d_k_proj, device=dev, dtype=dtype),
                torch.zeros(d_v_proj, device=dev, dtype=dtype),
                mixer.W_beta.bias,
                w_delta_b,
                mixer.W_out.bias
            ], dim=0))
            
        mixer.W_in = W_in
        mixer.is_fused = True
        
        # Monkey patch forward to use fused W_in
        def make_fused_forward(m):
            orig_forward = m.forward
            def fused_forward(x, state=None):
                B, T, _ = x.shape
                if not getattr(m, 'is_fused', False) or T > 1:
                    return orig_forward(x, state=state)
                
                # Single fused GEMM projection
                in_proj = m.W_in(x)  # [B, 1, total_dim]
                q, k, v, beta_raw, delta_raw, out_gate_raw = torch.split(
                    in_proj,
                    [m.H * m.d_k, m.H * m.d_k, m.H * m.d_v, m.H, m.H, m.H * m.d_v],
                    dim=-1
                )
                q = F.normalize(q.view(B, 1, m.H, m.d_k), p=2, dim=-1, eps=1e-6)
                k = F.normalize(k.view(B, 1, m.H, m.d_k), p=2, dim=-1, eps=1e-6)
                v = v.view(B, 1, m.H, m.d_v)
                out_gate = torch.sigmoid(out_gate_raw).view(B, 1, m.H, m.d_v)
                
                thresh = m.config.sparse_write_threshold
                beta = (F.relu(torch.sigmoid(beta_raw) - thresh) / (1.0 - thresh + 1e-8)) * torch.exp(m.log_beta_scale)[None, None, :]
                
                if m.config.use_variable_delta_t:
                    delta_t = F.softplus(delta_raw) + 1e-3
                    decay_rate = torch.exp(m.W_lambda)[None, None, :]
                    alpha = torch.exp(-delta_t * decay_rate).clamp(min=1e-4, max=0.9999)
                else:
                    alpha = torch.sigmoid(delta_raw).clamp(min=1e-4, max=0.9999)
                    
                S_prev, n_prev = state if state is not None else (None, None)
                S = S_prev if S_prev is not None else torch.zeros(B, m.H, m.d_k, m.d_v, device=x.device, dtype=x.dtype)
                n = n_prev if (m.config.use_epistemic_gate and n_prev is not None) else (
                    torch.zeros(B, m.H, m.d_k, device=x.device, dtype=x.dtype) if m.config.use_epistemic_gate else None
                )
                
                from kernel_nitro import fused_nitro_step
                out_step, S_new, n_new = fused_nitro_step(
                    q[:, 0].contiguous(), k[:, 0].contiguous(), v[:, 0].contiguous(),
                    alpha[:, 0].contiguous(), beta[:, 0].contiguous(), out_gate[:, 0].contiguous(),
                    S.contiguous(), n.contiguous() if n is not None else None
                )
                out = out_step.unsqueeze(1).reshape(B, 1, -1)
                return m.W_o(out), (S_new, n_new)
            return fused_forward
            
        mixer.forward = make_fused_forward(mixer)
    print(f"[Fused] Successfully fused all 6 linear projections into W_in across {len(model.layers)} layers.")


class CUDAGraphedHGDM:
    """
    Captures the single-token autoregressive generation step into a CUDA Graph,
    executing all 6 layers, fused projections, Triton step kernels, SwiGLU, and LM head
    as a single hardware pulse on the GPU stream.
    """
    def __init__(self, model: TokenizedHGDM, device: torch.device):
        self.model = model
        self.device = device
        self.config = model.config
        
        # 1. Static buffers
        self.static_token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.static_states = []
        for _ in range(model.config.n_layers):
            s = torch.zeros(1, model.config.n_heads, model.config.d_k, model.config.d_v, device=device, dtype=torch.bfloat16)
            n = torch.zeros(1, model.config.n_heads, model.config.d_k, device=device, dtype=torch.bfloat16) if model.config.use_epistemic_gate else None
            self.static_states.append((s, n))
            
        self.static_logits = torch.zeros(1, 1, model.config.vocab_size, device=device, dtype=torch.bfloat16)
        
        # 2. Warmup & Capture
        print("[CUDA Graph] Warming up and capturing graph for T=1 step...")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                logits, next_s = self.model(self.static_token, states=self.static_states)
                for i in range(len(self.model.layers)):
                    self.static_states[i][0].copy_(next_s[i][0])
                    if self.static_states[i][1] is not None:
                        self.static_states[i][1].copy_(next_s[i][1])
                self.static_logits.copy_(logits)
        torch.cuda.current_stream().wait_stream(s)
        
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=s):
            logits, next_s = self.model(self.static_token, states=self.static_states)
            for i in range(len(self.model.layers)):
                self.static_states[i][0].copy_(next_s[i][0])
                if self.static_states[i][1] is not None:
                    self.static_states[i][1].copy_(next_s[i][1])
            self.static_logits.copy_(logits)
            
        print("[CUDA Graph] Graph captured successfully!")

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 128) -> torch.Tensor:
        self.model.eval()
        B, T_prompt = prompt_ids.shape
        
        # 1. Prime recurrent state with prompt
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, prime_states = self.model(prompt_ids)
                
        # 2. Copy prime states into static graph state buffers
        for i in range(len(self.model.layers)):
            self.static_states[i][0].copy_(prime_states[i][0])
            if self.static_states[i][1] is not None:
                self.static_states[i][1].copy_(prime_states[i][1])
                
        curr_token = torch.argmax(logits[:, -1:], dim=-1, keepdim=True)
        generated = [prompt_ids, curr_token]
        
        # 3. Fast CUDA Graph replay loop
        for _ in range(max_new_tokens - 1):
            self.static_token.copy_(curr_token)
            self.graph.replay()
            # Fast greedy argmax directly from static_logits
            next_token = torch.argmax(self.static_logits[:, -1:], dim=-1)
            generated.append(next_token)
            curr_token = next_token
            
        return torch.cat(generated, dim=1)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
        
    device = torch.device("cuda")
    ckpt_path = "checkpoints_tokenized/ckpt_hgdm_sft.pt"
    data = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = data["config"]
    model = TokenizedHGDM(cfg).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()
    
    # 1. Fuse projections
    fuse_model_projections(model)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "User: What is artificial intelligence and how does it work?\nAssistant:"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    # 2. Build CUDA Graphed Runner
    graphed_model = CUDAGraphedHGDM(model, device)
    
    # 3. Benchmark 256 tokens
    N_TOKENS = 256
    print(f"\nBenchmarking {N_TOKENS} tokens with Fused GEMM + CUDA Graphs...")
    
    # Warmup
    _ = graphed_model.generate(prompt_ids, max_new_tokens=20)
    
    torch.cuda.synchronize()
    t0 = time.time()
    output_ids = graphed_model.generate(prompt_ids, max_new_tokens=N_TOKENS)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    
    new_tokens = output_ids.shape[1] - prompt_ids.shape[1]
    speed = new_tokens / elapsed
    
    resp = tokenizer.decode(output_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)
    
    print("\n" + "=" * 70)
    print("      FUSED GEMM + CUDA GRAPH SPEED DEMONSTRATION")
    print("=" * 70)
    print(f"Generated:  {new_tokens} tokens in {elapsed:.4f} seconds")
    print(f"Speed:      {speed:.1f} TOKENS / SECOND")
    print("=" * 70)
    print(f"Response Preview:\n{resp[:250]}...\n")

if __name__ == "__main__":
    main()
