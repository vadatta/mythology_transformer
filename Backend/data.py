import re
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tokenizer import hf_tokenizer
import random

BASE_DIR = Path(__file__).parent  # Backend/
DATA_DIR = BASE_DIR / "data"

def load_data(file_path):
    raw_text = Path(file_path).read_text(encoding="utf-8")
    entries = [x.strip() for x in re.split(r"\n\s*\n", raw_text) if x.strip()]
    return entries

def split_data(entries, split_ratio):
    entries = entries.copy()
    rng = random.Random()
    rng.shuffle(entries)

    split_idx = int(len(entries) * (1 - split_ratio))
    train_entries = entries[:split_idx]
    val_entries = entries[split_idx:]

    return train_entries, val_entries

class MythDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]

data = load_data('data/data.txt')
train_data, val_data = split_data(data, 0.2)

train_set = MythDataset(train_data)
val_set = MythDataset(val_data)


def collate(batch_data):
    encoded_batch = hf_tokenizer(batch_data, padding = True, truncation= True, max_length= 128, return_tensors= 'pt')

    input_ids = encoded_batch["input_ids"]
    attention_mask = encoded_batch["attention_mask"]

    #token shifting
    labels = input_ids.clone()
    labels[:, :-1] = labels[:, 1:]
    labels[:, -1] = -100

    #mask padding
    labels[:, :-1][attention_mask[:, 1:] == 0] = -100

    return {"input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask}




