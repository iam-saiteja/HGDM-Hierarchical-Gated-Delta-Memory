"""
CUDA Graph Autoregressive Generation Engine for HGDM.
Captures the single-token autoregressive recurrent step into a static CUDA Graph,
executing all layers, fused GEMM projections (W_in), fused Triton step kernels (TSG),
SwiGLU MLPs, and LM head projection as a single GPU pulse with zero CPU dispatch overhead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class CUDAGraphedHGDM:
    """
    Captures the single-token autoregressive generation step into a CUDA Graph,
    executing all layers, fused projections, Triton step kernels, SwiGLU, and LM head
    as a single hardware pulse on the GPU stream.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.config = getattr(model, "config", None)
        
        # Ensure model projections are fused
        if hasattr(self.model, "fuse_projections"):
            self.model.fuse_projections()
            
        # 1. Allocate static buffers on GPU
        self.static_token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.static_states: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []
        for layer in self.model.layers:
            if hasattr(layer.mixer, "init_state"):
                s = layer.mixer.init_state(1, device, dtype=torch.bfloat16)
            else:
                s_mat = torch.zeros(
                    1, self.config.n_heads, self.config.d_k, self.config.d_v,
                    device=device, dtype=torch.bfloat16
                )
                n_vec = (
                    torch.zeros(1, self.config.n_heads, self.config.d_k, device=device, dtype=torch.bfloat16)
                    if self.config.use_epistemic_gate else None
                )
                s = (s_mat, n_vec)
            self.static_states.append(s)
            
        self.static_logits = torch.zeros(
            1, 1, self.config.vocab_size, device=device, dtype=torch.bfloat16
        )
        
        # 2. Warmup & Capture CUDA Graph
        print("[CUDA Graph] Warming up and capturing graph for T=1 step...")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                for _ in range(5):
                    logits, next_s = self.model(self.static_token, states=self.static_states)
                    for i in range(len(self.model.layers)):
                        self.static_states[i][0].copy_(next_s[i][0])
                        if self.static_states[i][1] is not None and next_s[i][1] is not None:
                            self.static_states[i][1].copy_(next_s[i][1])
                        if len(self.static_states[i]) > 2 and len(next_s[i]) > 2:
                            if self.static_states[i][2] is not None and next_s[i][2] is not None:
                                self.static_states[i][2].copy_(next_s[i][2])
                    self.static_logits.copy_(logits)
        torch.cuda.current_stream().wait_stream(s)
        
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=s):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, next_s = self.model(self.static_token, states=self.static_states)
                for i in range(len(self.model.layers)):
                    self.static_states[i][0].copy_(next_s[i][0])
                    if self.static_states[i][1] is not None and next_s[i][1] is not None:
                        self.static_states[i][1].copy_(next_s[i][1])
                    if len(self.static_states[i]) > 2 and len(next_s[i]) > 2:
                        if self.static_states[i][2] is not None and next_s[i][2] is not None:
                            self.static_states[i][2].copy_(next_s[i][2])
                self.static_logits.copy_(logits)
            
        print("[CUDA Graph] Graph captured successfully!")

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        self.model.eval()
        B, T_prompt = prompt_ids.shape
        assert B == 1, f"CUDAGraphedHGDM currently supports batch_size=1, got {B}"
        
        # 1. Prime recurrent state with prompt
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, prime_states = self.model(prompt_ids)
            
        # 2. Copy prime states into static graph state buffers
        for i in range(len(self.model.layers)):
            self.static_states[i][0].copy_(prime_states[i][0])
            if self.static_states[i][1] is not None and prime_states[i][1] is not None:
                self.static_states[i][1].copy_(prime_states[i][1])
            if len(self.static_states[i]) > 2 and len(prime_states[i]) > 2:
                if self.static_states[i][2] is not None and prime_states[i][2] is not None:
                    self.static_states[i][2].copy_(prime_states[i][2])
                
        if temperature > 0:
            logits_step = logits[:, -1] / max(1e-5, temperature)
            if top_k > 0:
                v, _ = torch.topk(logits_step, min(top_k, logits_step.size(-1)))
                logits_step[logits_step < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits_step, dim=-1)
            curr_token = torch.multinomial(probs, num_samples=1)
        else:
            curr_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            
        generated = [prompt_ids, curr_token]
        
        # 3. Fast CUDA Graph replay loop
        for _ in range(max_new_tokens - 1):
            self.static_token.copy_(curr_token)
            self.graph.replay()
            
            if temperature > 0:
                logits_step = self.static_logits[:, -1] / max(1e-5, temperature)
                if top_k > 0:
                    v, _ = torch.topk(logits_step, min(top_k, logits_step.size(-1)))
                    logits_step[logits_step < v[:, [-1]]] = -float('inf')
                probs = F.softmax(logits_step, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(self.static_logits[:, -1], dim=-1, keepdim=True)
                
            generated.append(next_token)
            curr_token = next_token
            
        return torch.cat(generated, dim=1)
