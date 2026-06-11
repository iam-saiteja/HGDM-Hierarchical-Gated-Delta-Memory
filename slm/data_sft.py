import torch
from datasets import load_dataset

# ChatML standard tags as byte arrays
IM_START = b"<|im_start|>"
IM_END = b"<|im_end|>\n"
USER_ROLE = b"user\n"
ASSISTANT_ROLE = b"assistant\n"
SYSTEM_ROLE = b"system\n"

def format_chat_to_bytes_and_targets(messages, max_length=2048):
    """
    Takes a list of message dicts: [{'role': 'user', 'content': '...'}, ...]
    Returns input byte tensor `x` and target byte tensor `y`.
    `y` has -100 for bytes belonging to the user/system prompts.
    """
    byte_stream = []
    target_stream = []
    
    for msg in messages:
        role = msg['role']
        content = msg['content'].encode('utf-8', errors='replace')
        
        if role == 'user':
            role_bytes = USER_ROLE
        elif role == 'assistant':
            role_bytes = ASSISTANT_ROLE
        else:
            role_bytes = SYSTEM_ROLE
            
        header = IM_START + role_bytes
        
        # Append header (we do not calculate loss on the headers)
        byte_stream.extend(header)
        target_stream.extend([-100] * len(header))
        
        if role == 'assistant':
            # Target for assistant content is the content itself
            byte_stream.extend(content)
            target_stream.extend(content)
            
            # Predict the end tag as well
            byte_stream.extend(IM_END)
            target_stream.extend(IM_END)
        else:
            # User or System. We do not predict this.
            byte_stream.extend(content)
            target_stream.extend([-100] * len(content))
            
            byte_stream.extend(IM_END)
            target_stream.extend([-100] * len(IM_END))
            
    # Convert to tensors
    seq = torch.tensor(byte_stream, dtype=torch.long)
    tgt = torch.tensor(target_stream, dtype=torch.long)
    
    if len(seq) > max_length + 1:
        # truncate
        seq = seq[:max_length+1]
        tgt = tgt[:max_length+1]
        
    x = seq[:-1]
    y = tgt[1:]
    
    # pad if needed
    if len(x) < max_length:
        pad_len = max_length - len(x)
        x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
        y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])
        
    return x, y

def get_sft_dataloader(dataset_name="HuggingFaceH4/ultrachat_200k", split="train_sft", batch_size=4, block_size=2048):
    """
    Streams a conversational dataset and yields batches of (x, y).
    Automatically maps different dataset formats (e.g. HelpSteer vs Ultrachat) to standard messages list.
    """
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    def data_generator():
        x_batch = []
        y_batch = []
        for row in dataset:
            messages = []
            
            # Format mapping
            if 'messages' in row:
                # E.g. Ultrachat format
                messages = row['messages']
            elif 'prompt' in row and 'response' in row:
                # E.g. HelpSteer2 format
                messages = [
                    {'role': 'user', 'content': row['prompt']},
                    {'role': 'assistant', 'content': row['response']}
                ]
            else:
                continue # Skip unknown formats
                
            x, y = format_chat_to_bytes_and_targets(messages, max_length=block_size)
            
            # Skip if the target is entirely -100 (e.g., if truncation cut off the assistant response)
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
    # Test the loader
    print("Testing SFT Dataloader (HelpSteer2)...")
    loader = get_sft_dataloader("nvidia/HelpSteer2", split="train", batch_size=2, block_size=128)
    data_stream = iter(loader)
    x, y = next(data_stream)
    
    print(f"Batch X Shape: {x.shape}")
    print(f"Batch Y Shape: {y.shape}")
    
    # Verify the first sequence formatting
    bytes_x = bytes(x[0].tolist())
    print(f"\nDecoded X (Sample 0):\n{bytes_x.decode('utf-8', errors='replace')}")
    
    # Print targets for the first 100 bytes to show the -100 masking
    print("\nTarget array (Y) for Sample 0:")
    print(y[0].tolist()[:100])
