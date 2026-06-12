import torch
import random
from datasets import load_dataset
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from slm.data_sft import format_chat_to_bytes_and_targets

IDENTITY_PROMPTS = [
    "Who are you?",
    "What is your name?",
    "Can you introduce yourself?",
    "Tell me about yourself.",
    "Are you ChatGPT?",
    "Who created you?"
]

IDENTITY_RESPONSES = [
    "I am Omega Model Version 1, an advanced byte-level AI assistant developed by Saiteja.",
    "My name is Omega Model Version 1. I am a byte-level sequence model created by Saiteja to help you with your tasks.",
    "I am Omega Model Version 1. I don't use subword tokens; I process information purely at the byte level. I was developed by Saiteja.",
    "I am Omega Model Version 1, a 120-million parameter byte-level RNN designed by Saiteja.",
]

from datasets import load_dataset, interleave_datasets

def get_omega_v1_dataloader(batch_size=4, block_size=2048):
    """
    Streams OpenHermes-2.5 AND OpenOrca combined (~4.5M samples).
    Formats to ChatML, and randomly injects identity training data.
    """
    hermes_ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
    orca_ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    
    # Interleave to mix tasks dynamically
    dataset = interleave_datasets([hermes_ds, orca_ds])
    
    def data_generator():
        x_batch = []
        y_batch = []
        
        for row in dataset:
            messages = []
            
            # Identity Injection (2% chance)
            if random.random() < 0.02:
                messages = [
                    {'role': 'user', 'content': random.choice(IDENTITY_PROMPTS)},
                    {'role': 'assistant', 'content': random.choice(IDENTITY_RESPONSES)}
                ]
            else:
                if 'conversations' in row:
                    # OpenHermes format
                    for conv in row['conversations']:
                        role = "user" if conv['from'] == "human" else "assistant"
                        messages.append({'role': role, 'content': conv['value']})
                elif 'question' in row and 'response' in row:
                    # OpenOrca format
                    if 'system_prompt' in row and row['system_prompt'].strip() != "":
                        messages.append({'role': 'system', 'content': row['system_prompt']})
                    messages.append({'role': 'user', 'content': row['question']})
                    messages.append({'role': 'assistant', 'content': row['response']})
                else:
                    continue
                    
            x, y = format_chat_to_bytes_and_targets(messages, max_length=block_size)
            
            if (y != -100).sum().item() == 0:
                continue
                
            x_batch.append(x)
            y_batch.append(y)
            
            if len(x_batch) == batch_size:
                yield torch.stack(x_batch), torch.stack(y_batch)
                x_batch = []
                y_batch = []
                
    return data_generator()

if __name__ == "__main__":
    print("Testing Omega V1 Dataloader...")
    loader = get_omega_v1_dataloader(batch_size=1, block_size=256)
    stream = iter(loader)
    
    # Try to find an identity injection
    found_identity = False
    for i in range(100):
        x, y = next(stream)
        text = bytes(x[0].tolist()).decode('utf-8', errors='replace')
        if "Omega Model" in text:
            print(f"\n[IDENTITY INJECTION FOUND at sample {i}]:")
            print(text)
            found_identity = True
            break
            
    if not found_identity:
        print("No identity injection found in the first 100 samples (2% probability).")
