
import sentencepiece as spm
import os

CORPUS_FILE = "sp_corpus.txt"
MODEL_PREFIX = "token"
VOCAB_SIZE = 2000

# write corpus once
if not os.path.exists(CORPUS_FILE):
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        f.write(text)

# train tokenizer once
if not os.path.exists(f"{MODEL_PREFIX}.model"):
    spm.SentencePieceTrainer.Train(
        input=CORPUS_FILE,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe"
    )

sp = spm.SentencePieceProcessor()
sp.Load(f"{MODEL_PREFIX}.model")

def encode(s):
    return sp.Encode(s, out_type=int)

def decode(tokens):
    return sp.Decode(tokens)
