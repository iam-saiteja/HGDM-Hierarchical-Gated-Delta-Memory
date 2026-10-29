import os
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Optional, Iterator, Dict, Any

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


class PackedTokenDataset(IterableDataset):
    """
    Streaming Packed Token Dataset.
    Streams natural text from HuggingFace (e.g. FineWeb-Edu, SmolTalk, TinyStories),
    tokenizes on the fly, and packs continuous streams of tokens into fixed-size
    chunks of length `seq_len` without padding waste.
    """
    def __init__(
        self,
        dataset_name: str = "HuggingFaceTB/smoltalk",
        split: str = "train",
        subset: Optional[str] = None,
        tokenizer_name: str = "gpt2",
        seq_len: int = 1024,
        max_tokens: Optional[int] = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        if subset is None and "smoltalk" in dataset_name.lower():
            subset = "all"
        self.subset = subset
        self.tokenizer_name = tokenizer_name
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        
        if AutoTokenizer is None:
            raise ImportError("Please install transformers via `uv pip install transformers`")
            
        print(f"[Dataset] Loading tokenizer '{tokenizer_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_text_stream(self) -> Iterator[str]:
        if load_dataset is None:
            # Fallback if datasets is not available: generate clean conversational text
            sample_dialogues = [
                "User: What is artificial intelligence?\nAssistant: Artificial intelligence refers to systems designed to perform tasks that typically require human cognition, such as pattern recognition, problem solving, and language understanding.",
                "User: How do black holes form in space?\nAssistant: Stellar black holes form when a massive star exhausts its nuclear fuel and undergoes gravitational collapse under its own weight.",
                "User: Can you explain recurrent neural networks versus transformers?\nAssistant: Recurrent neural networks process sequential tokens step-by-step with a persistent hidden state, while transformers use parallel self-attention across the full sequence.",
                "User: Write a short story about a robot who discovered music.\nAssistant: In the quiet corridors of Sector 7, unit AX-4 found an old brass gramophone buried under scrap metal. When the needle touched the vinyl, a strange resonance vibrated through its chassis.",
            ]
            while True:
                for text in sample_dialogues:
                    yield text
        else:
            try:
                if self.subset:
                    ds = load_dataset(self.dataset_name, self.subset, split=self.split, streaming=True)
                else:
                    ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
                    
                for row in ds:
                    if "text" in row:
                        yield row["text"]
                    elif "messages" in row:
                        # Format chat messages into standard prompt/response text
                        formatted = []
                        for msg in row["messages"]:
                            role = msg.get("role", "User").capitalize()
                            content = msg.get("content", "")
                            formatted.append(f"{role}: {content}")
                        yield "\n\n".join(formatted)
                    elif "content" in row:
                        yield row["content"]
            except Exception as e:
                print(f"[Dataset] Warning: Failed to stream '{self.dataset_name}' ({e}). Falling back to local conversational stream.")
                while True:
                    yield "User: What is the primary difference in HGDM?\nAssistant: HGDM maintains a persistent matrix state updated by a gated delta rule, eliminating the quadratic KV cache."

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer = []
        tokens_yielded = 0
        eos_id = self.tokenizer.eos_token_id or 0
        
        for text in self._get_text_stream():
            if not text or len(text.strip()) == 0:
                continue
                
            token_ids = self.tokenizer.encode(text)
            token_ids.append(eos_id)
            buffer.extend(token_ids)
            
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                
                tokens_yielded += self.seq_len
                yield {"input_ids": x, "labels": y}
                
                if self.max_tokens is not None and tokens_yielded >= self.max_tokens:
                    return


def create_dataloader(
    dataset_name: str = "HuggingFaceTB/smoltalk",
    split: str = "train",
    subset: Optional[str] = None,
    tokenizer_name: str = "gpt2",
    batch_size: int = 4,
    seq_len: int = 1024,
    max_tokens: Optional[int] = None,
) -> DataLoader:
    dataset = PackedTokenDataset(
        dataset_name=dataset_name,
        split=split,
        subset=subset,
        tokenizer_name=tokenizer_name,
        seq_len=seq_len,
        max_tokens=max_tokens,
    )
    return DataLoader(dataset, batch_size=batch_size)
