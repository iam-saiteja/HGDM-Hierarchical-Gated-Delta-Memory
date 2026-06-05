import torch
import torch.nn as nn
import torch.nn.functional as F
from ultimate.hgdm_ultimate import HGDMUltimate, HGDMConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

def test_details():
    config = HGDMConfig(use_rope=False, use_epistemic_gate=True, n_grad_mode="exact")
    
    # 1. Instantiate both models
    model_seq = HGDMUltimate(config, force_sequential=True).to(DEVICE)
    model_tri = HGDMUltimate(config, force_sequential=False).to(DEVICE)
    
    # Load same weights
    model_tri.load_state_dict(model_seq.state_dict())
    
    # Put in eval mode to disable dropout etc
    model_seq.eval()
    model_tri.eval()
    
    # Input
    B, T = 2, 64
    x = torch.randint(0, 256, (B, T), device=DEVICE)
    
    # Run embedding
    x_emb_seq = model_seq.embedding(x)
    x_emb_tri = model_tri.embedding(x)
    print("Embedding diff:", (x_emb_seq - x_emb_tri).abs().max().item())
    
    # Let's run layer by layer
    curr_seq = x_emb_seq
    curr_tri = x_emb_tri
    
    states_seq = [None] * len(model_seq.layers)
    states_tri = [None] * len(model_tri.layers)
    
    boundary_mask = torch.zeros_like(x, dtype=torch.bool)
    for token_id in config.boundary_token_ids:
        boundary_mask = boundary_mask | (x == token_id)
        
    for l_idx in range(config.n_layers):
        layer_seq = model_seq.layers[l_idx]
        layer_tri = model_tri.layers[l_idx]
        
        # Norm1
        norm_seq = layer_seq.norm1(curr_seq)
        norm_tri = layer_tri.norm1(curr_tri)
        print(f"\nLayer {l_idx} Norm1 diff:", (norm_seq - norm_tri).abs().max().item())
        
        # Mixer
        m_seq, ns_seq = layer_seq.mixer(norm_seq, state=states_seq[l_idx], boundary_mask=boundary_mask)
        m_tri, ns_tri = layer_tri.mixer(norm_tri, state=states_tri[l_idx], boundary_mask=boundary_mask)
        
        print(f"Layer {l_idx} Mixer output diff:", (m_seq - m_tri).abs().max().item())
        
        S_seq, n_seq = ns_seq
        S_tri, n_tri = ns_tri
        
        print(f"Layer {l_idx} State S diff:", (S_seq - S_tri).abs().max().item())
        print(f"Layer {l_idx} State n diff:", (n_seq - n_tri).abs().max().item())
        
        # We can also compare current outputs
        curr_seq_next = curr_seq + m_seq
        curr_seq_next = curr_seq_next + layer_seq.ffn(layer_seq.norm2(curr_seq_next))
        
        curr_tri_next = curr_tri + m_tri
        curr_tri_next = curr_tri_next + layer_tri.ffn(layer_tri.norm2(curr_tri_next))
        
        print(f"Layer {l_idx} FFN input diff:", ((curr_seq + m_seq) - (curr_tri + m_tri)).abs().max().item())
        print(f"Layer {l_idx} FFN output diff:", (layer_seq.ffn(layer_seq.norm2(curr_seq_next)) - layer_tri.ffn(layer_tri.norm2(curr_tri_next))).abs().max().item())
        print(f"Layer {l_idx} Final layer output diff:", (curr_seq_next - curr_tri_next).abs().max().item())
        
        curr_seq = curr_seq_next
        curr_tri = curr_tri_next
        
    # Final Norm and FC
    final_seq = model_seq.norm_f(curr_seq)
    final_tri = model_tri.norm_f(curr_tri)
    print("\nFinal norm diff:", (final_seq - final_tri).abs().max().item())
    
    out_seq = model_seq.fc_out(final_seq)
    out_tri = model_tri.fc_out(final_tri)
    print("Final logit diff:", (out_seq - out_tri).abs().max().item())

if __name__ == "__main__":
    test_details()
