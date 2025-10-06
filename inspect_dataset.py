import torch
from transformers import AutoTokenizer

# -----------------------------
# Load the dataset safely
# -----------------------------
encodings = torch.load("sample_dataset.pt", weights_only=False)

print("✅ Loaded sample_dataset.pt")
print("Type:", type(encodings))
print("Keys in dataset:", encodings.keys())
print("Shape of input_ids:", encodings["input_ids"].shape)
print("Shape of attention_mask:", encodings["attention_mask"].shape)

# -----------------------------
# Initialize tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# -----------------------------
# Decode all sequences
# -----------------------------
print("\n🔎 Decoded Sequences:\n")

all_input_ids = encodings["input_ids"]

for i in range(20):
    decoded_text = tokenizer.decode(all_input_ids[i], skip_special_tokens=True)
    print(f"[{i+1}] {decoded_text}")
