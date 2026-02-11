import random
from pathlib import Path
import torch
from .tokenizer import encode

BASE_DIR = Path(__file__).parent  # Backend/
DATA_DIR = BASE_DIR / "data"

with open(DATA_DIR / "Trojan_War.txt") as f:
    text1 = f.read()

with open(DATA_DIR / "Greek_mythology.txt") as f:
    text2 = f.read()

with open(DATA_DIR / "Odyssey.txt") as f:
    text3 = f.read()

with open(DATA_DIR / "mythology_qa.txt") as f:
    qa_text= f.read()

text = text1 + "\n\n" + text2 + "\n\n" + text3

#split text into blocks of length 200 for randomization
def split_blocks(text, min_len=200):
    blocks = [
        b.strip()
        for b in text.split("\n\n")
        if len(b.strip()) >= min_len
    ]
    return blocks

#randomization
def build_training_text():
    blocks = (
            split_blocks(text1) +
            split_blocks(text2) +
            split_blocks(text3)
    )

    random.shuffle(blocks)

    text =  "\n\n".join(blocks)
    return torch.tensor(encode(text), dtype=torch.long)
