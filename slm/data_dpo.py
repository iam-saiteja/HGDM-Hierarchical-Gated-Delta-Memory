import torch
from datasets import load_dataset
from slm.data_sft import format_chat_to_bytes_and_targets

def get_dpo_dataloader(dataset_name="nvidia/HelpSteer2", split="train", batch_size=2, block_size=2048):
    """
    Streams a preference dataset (HelpSteer2) and yields:
    (x_chosen, y_chosen, x_rejected, y_rejected)
    """
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    def data_generator():
        xc_batch, yc_batch = [], []
        xr_batch, yr_batch = [], []
        
        # Buffer to find the highest and lowest reward responses for a given prompt
        # HelpSteer2 provides multiple responses per prompt
        prompt_buffer = {}
        
        for row in dataset:
            prompt = row['prompt']
            response = row['response']
            
            # Calculate a synthetic reward based on HelpSteer annotations
            # Sum of helpfulness, coherence, complexity, factuality minus verbosity penalty
            reward = row.get('helpfulness', 0) + row.get('coherence', 0) + row.get('factuality', 0)
            
            if prompt not in prompt_buffer:
                prompt_buffer[prompt] = []
            prompt_buffer[prompt].append((response, reward))
            
            # If we have at least 2 responses for this prompt, we can form a pair
            if len(prompt_buffer[prompt]) >= 2:
                # Sort by reward
                prompt_buffer[prompt].sort(key=lambda x: x[1], reverse=True)
                
                chosen_response = prompt_buffer[prompt][0][0]
                rejected_response = prompt_buffer[prompt][-1][0]
                
                # Format Chosen
                chosen_msgs = [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': chosen_response}
                ]
                xc, yc = format_chat_to_bytes_and_targets(chosen_msgs, max_length=block_size)
                
                # Format Rejected
                rejected_msgs = [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': rejected_response}
                ]
                xr, yr = format_chat_to_bytes_and_targets(rejected_msgs, max_length=block_size)
                
                # Ensure they fit and are valid
                if (yc != -100).sum().item() > 0 and (yr != -100).sum().item() > 0:
                    xc_batch.append(xc)
                    yc_batch.append(yc)
                    xr_batch.append(xr)
                    yr_batch.append(yr)
                
                # Clear buffer for this prompt
                del prompt_buffer[prompt]
            
            if len(xc_batch) == batch_size:
                yield torch.stack(xc_batch), torch.stack(yc_batch), torch.stack(xr_batch), torch.stack(yr_batch)
                xc_batch, yc_batch = [], []
                xr_batch, yr_batch = [], []
                
    return data_generator()

if __name__ == "__main__":
    print("Testing DPO Dataloader...")
    loader = get_dpo_dataloader(batch_size=1, block_size=512)
    stream = iter(loader)
    xc, yc, xr, yr = next(stream)
    print("Chosen shape:", xc.shape)
    print("Rejected shape:", xr.shape)
