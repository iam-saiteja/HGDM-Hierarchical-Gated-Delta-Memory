import torch
import torch.nn.functional as F
import argparse
import sys
import os
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hgdm_omega import OmegaGDM, OmegaConfig

def get_choice_loss(model, context_bytes, choice_bytes, device):
    """Calculates the cross-entropy loss of the choice_bytes given the context."""
    full_seq = context_bytes + choice_bytes
    x = torch.tensor([full_seq[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([full_seq[1:]], dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits, _ = model(x)
        
    # We only care about the loss over the choice bytes
    # The choice starts at index len(context_bytes) in the sequence
    # In the labels y, the first choice byte is at len(context_bytes) - 1
    shift = len(context_bytes) - 1
    logits_choice = logits[0, shift:]
    y_choice = y[0, shift:]
    
    loss = F.cross_entropy(logits_choice, y_choice, reduction='sum')
    return loss.item()

def evaluate_boolq(model, device, num_samples=200):
    print("\n--- Evaluating BoolQ (Zero-Shot) ---")
    dataset = load_dataset("boolq", split="validation", trust_remote_code=True)
    correct = 0
    total = 0
    
    for i in tqdm(range(min(num_samples, len(dataset)))):
        item = dataset[i]
        passage = item['passage']
        question = item['question']
        answer = item['answer'] # True or False
        
        prompt = f"<|im_start|>user\nPassage: {passage}\nQuestion: {question}?\nAnswer Yes or No.<|im_end|>\n<|im_start|>assistant\nAnswer:"
        
        c_bytes = list(prompt.encode('utf-8'))
        yes_bytes = list(" Yes".encode('utf-8'))
        no_bytes = list(" No".encode('utf-8'))
        
        loss_yes = get_choice_loss(model, c_bytes, yes_bytes, device)
        loss_no = get_choice_loss(model, c_bytes, no_bytes, device)
        
        pred = True if loss_yes < loss_no else False
        if pred == answer:
            correct += 1
        total += 1
        
    acc = correct / total
    print(f"BoolQ Accuracy: {acc*100:.2f}%")
    return acc

def evaluate_arc_easy(model, device, num_samples=200):
    print("\n--- Evaluating ARC-Easy (Zero-Shot) ---")
    dataset = load_dataset("ai2_arc", "ARC-Easy", split="validation", trust_remote_code=True)
    correct = 0
    total = 0
    
    for i in tqdm(range(min(num_samples, len(dataset)))):
        item = dataset[i]
        question = item['question']
        choices = item['choices']['text']
        labels = item['choices']['label']
        answer_key = item['answerKey']
        
        prompt = f"<|im_start|>user\nQuestion: {question}\nChoices:\n"
        for label, text in zip(labels, choices):
            prompt += f"{label}: {text}\n"
        prompt += f"Correct Answer:<|im_end|>\n<|im_start|>assistant\n"
        
        c_bytes = list(prompt.encode('utf-8'))
        
        best_loss = float('inf')
        best_pred = None
        
        for label, text in zip(labels, choices):
            choice_bytes = list(f"{label}".encode('utf-8'))
            loss = get_choice_loss(model, c_bytes, choice_bytes, device)
            if loss < best_loss:
                best_loss = loss
                best_pred = label
                
        if best_pred == answer_key:
            correct += 1
        total += 1
        
    acc = correct / total
    print(f"ARC-Easy Accuracy: {acc*100:.2f}%")
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="omega_v1_dpo_latest.pt")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading OmegaGDM on {device}...")
    
    cfg = OmegaConfig(
        d_byte=256, catcher_layers=2, renderer_layers=2,
        d_model=768, core_layers=12, n_heads=12,
        d_k=64, d_v=64, d_ff=3072,
        decimation_rate=8, max_position_embeddings=2048,
        vocab_size=256, use_state_fusion=False
    )
    
    model = OmegaGDM(cfg, force_sequential=False).to(device)
    
    if not os.path.exists(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}")
        sys.exit(1)
        
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    
    # Run targeted evaluations to compare against KAI-95M
    acc_boolq = evaluate_boolq(model, device, num_samples=250)
    acc_arc = evaluate_arc_easy(model, device, num_samples=250)
    
    print("\n=========================================")
    print("FINAL BENCHMARK RESULTS")
    print("=========================================")
    print(f"OmegaGDM (120M) BoolQ : {acc_boolq*100:.1f}% | KAI-95M: 62.0% | GPT-2: 48.7%")
    print(f"OmegaGDM (120M) ARC-E : {acc_arc*100:.1f}% | KAI-95M: 26.6% | GPT-2: 43.8%")
    print("=========================================")

if __name__ == "__main__":
    main()
