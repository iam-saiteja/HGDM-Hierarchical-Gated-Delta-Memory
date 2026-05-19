import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

class MultiSourceChunkedByteDataset(IterableDataset):
    def __init__(self, block_size=2048, fineweb_ratio=0.6, wiki_ratio=0.25, code_ratio=0.15):
        super().__init__()
        self.block_size = block_size
        self.ratios = torch.tensor([fineweb_ratio, wiki_ratio, code_ratio])
        
        # Stream sources directly from Hugging Face (using training splits)
        print("[Dataset] Initializing streaming data pipelines (Training Splits)...")
        self.fineweb = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        self.wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        self.code = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)

    def __iter__(self):
        fw_iter = iter(self.fineweb)
        wiki_iter = iter(self.wiki)
        code_iter = iter(self.code)
        
        buffer = []
        
        while True:
            # Randomly select a source based on defined proportions
            choice = torch.multinomial(self.ratios, 1).item()
            try:
                if choice == 0:
                    text = next(fw_iter)['text']
                elif choice == 1:
                    text = next(wiki_iter)['text']
                else:
                    text = next(code_iter)['content']
                
                # Encode straight to raw UTF-8 integer bytes
                raw_bytes = list(text.encode('utf-8', errors='ignore'))
                buffer.extend(raw_bytes)
                
                # Buffer growth guard: cap buffer size to avoid potential memory leak on giant documents
                if len(buffer) > 10_000_000:
                    buffer = buffer[-(self.block_size + 1):]
                
                # Drain the buffer into fixed-size context blocks
                while len(buffer) >= self.block_size + 1:
                    chunk = buffer[:self.block_size + 1]
                    buffer = buffer[self.block_size:]
                    yield torch.tensor(chunk, dtype=torch.long)
                    
            except StopIteration:
                # Re-initialize the exhausted iterator to keep stream alive smoothly
                if choice == 0: fw_iter = iter(self.fineweb)
                elif choice == 1: wiki_iter = iter(self.wiki)
                else: code_iter = iter(self.code)
                continue

def get_1b_dataloader(block_size=2048, batch_size=2):
    dataset = MultiSourceChunkedByteDataset(block_size=block_size)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
