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


class StatefulStreamDataset:
    """
    Maintains B parallel token streams for stateful recurrent training (TBPTT).
    Each stream advances continuously through documents.
    Returns:
      input_ids: [B, seq_len]
      labels: [B, seq_len]
      reset_mask: [B] (True if this chunk is the start of a new document, requiring state zeroing)
    """
    def __init__(
        self,
        dataset_name: str = "HuggingFaceTB/smoltalk",
        split: str = "train",
        subset: Optional[str] = None,
        tokenizer_name: str = "gpt2",
        batch_size: int = 4,
        seq_len: int = 1024,
        max_tokens: Optional[int] = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        if subset is None and "smoltalk" in dataset_name.lower():
            subset = "all"
        self.subset = subset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        
        if AutoTokenizer is None:
            raise ImportError("Please install transformers via `uv pip install transformers`")
            
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id or 0
        
        # Instantiate B independent document iterators
        self.streams = [
            self._create_document_stream(self.dataset_name, self.split, self.subset, stream_idx=i)
            for i in range(batch_size)
        ]
        self.buffers = [[] for _ in range(batch_size)]
        self.is_new_doc = [True for _ in range(batch_size)]
        self.tokens_yielded = 0

    def _create_document_stream(self, dataset_name, split, subset, stream_idx=0):
        if load_dataset is None:
            sample_dialogues = [
                "User: What is artificial intelligence?\nAssistant: Artificial intelligence refers to systems designed to perform cognitive tasks.",
                "User: Explain continuous recurrent memory.\nAssistant: HGDM maintains a persistent state matrix across tokens.",
                "User: Tell me a story about exploration.\nAssistant: The rover rolled across the red dust of the crater, searching for ancient water.",
            ]
            while True:
                for s in sample_dialogues:
                    yield s
        else:
            try:
                if subset:
                    ds = load_dataset(dataset_name, subset, split=split, streaming=True)
                else:
                    ds = load_dataset(dataset_name, split=split, streaming=True)
                for i, row in enumerate(ds):
                    if i < stream_idx:
                        continue
                    if "messages" in row:
                        formatted = []
                        for msg in row["messages"]:
                            role = msg.get("role", "User").capitalize()
                            content = msg.get("content", "")
                            formatted.append(f"{role}: {content}")
                        yield "\n\n".join(formatted)
                    elif "text" in row:
                        yield row["text"]
                    elif "content" in row:
                        yield row["content"]
            except Exception as e:
                while True:
                    yield "User: What is HGDM?\nAssistant: HGDM provides O(1) stateful recurrence with hierarchical temporal bands."

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self.max_tokens is not None and self.tokens_yielded >= self.max_tokens:
            raise StopIteration
            
        batch_inputs = []
        batch_labels = []
        reset_mask = []
        
        for b in range(self.batch_size):
            resets_this_slot = False
            while len(self.buffers[b]) < self.seq_len + 1:
                try:
                    doc = next(self.streams[b])
                except StopIteration:
                    self.streams[b] = self._create_document_stream(
                        self.dataset_name, self.split, self.subset, stream_idx=b
                    )
                    doc = next(self.streams[b])
                    
                tokens = self.tokenizer.encode(doc)
                tokens.append(self.eos_id)
                
                if len(self.buffers[b]) == 0:
                    resets_this_slot = True
                self.buffers[b].extend(tokens)
                
            chunk = self.buffers[b][:self.seq_len + 1]
            self.buffers[b] = self.buffers[b][self.seq_len:]
            
            x = chunk[:-1]
            y = chunk[1:]
            
            batch_inputs.append(x)
            batch_labels.append(y)
            reset_mask.append(resets_this_slot)
            
        self.tokens_yielded += self.batch_size * self.seq_len
        
        return {
            "input_ids": torch.tensor(batch_inputs, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "reset_mask": torch.tensor(reset_mask, dtype=torch.bool),
        }
