import pandas as pd
import re
from transformers import AutoTokenizer
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torch

# -----------------------------
# Step 1: Convert row -> text
# -----------------------------
def row_to_text(row):
    """
    Convert one row into a natural language sentence for the text corpus.
    """
    return (
        f"On {row['event_time']}, user {row['user_id']} "
        f"{row['event_type']} a {row['brand']} product in category {row['category_id']} "
        f"priced at {row['price']} during session {row['user_session']}."
    )

# -----------------------------
# Step 2: Clean text
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)          # remove extra spaces
    text = re.sub(r'[^a-z0-9\s.,]', '', text) # remove strange symbols
    return text.strip()

# -----------------------------
# Step 3: Stream CSV in chunks
# -----------------------------
def stream_csv(path, tokenizer, chunk_size=50000, block_size=128):
    """
    Read CSV in chunks to avoid memory errors.
    Tokenize each chunk and yield PyTorch encodings.
    """
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        texts = [
            clean_text(row_to_text(row))
            for _, row in chunk.iterrows()
            if pd.notnull(row["brand"]) and len(str(row["brand"])) > 1
        ]
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=block_size,
            return_tensors="pt"
        )
        yield encodings

# -----------------------------
# Step 4: Custom PyTorch Dataset
# -----------------------------
class ClickstreamDataset(TorchDataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }

# -----------------------------
# Example Run
# -----------------------------
if __name__ == "__main__":
    csv_path = r"C:\Users\mtris\OneDrive\Desktop\MS Admissions Spring 2024\Universities\Northeastern University\Course materials\4th semester\CSYE 7374\archive\2019-Nov.csv"

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    for i, encodings in enumerate(stream_csv(csv_path, tokenizer, chunk_size=50000)):
        dataset = ClickstreamDataset(encodings)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        print(f"✅ Processed chunk {i+1}, rows in this chunk: {len(dataset)}")

        # Save only the first chunk as sample for submission
        if i == 0:
            torch.save(encodings, "sample_dataset.pt")
            print("💾 Saved tokenized sample batch -> sample_dataset.pt")
            break
