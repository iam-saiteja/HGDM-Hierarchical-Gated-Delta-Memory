from datasets import load_dataset
try:
    ds = load_dataset("Helsinki-NLP/opus_books", "en-es", split="train")
    print("Dataset size:", len(ds))
except Exception as e:
    print("Error:", e)
