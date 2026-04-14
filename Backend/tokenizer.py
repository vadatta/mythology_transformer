from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from transformers import PreTrainedTokenizerFast

CORPUS_FILE = "data/data.txt"
VOCAB_SIZE = 1000


tokenizer_hf = Tokenizer(models.BPE(unk_token="[UNK]"))

tokenizer_hf.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase()])

tokenizer_hf.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
)

tokenizer_hf.train([CORPUS_FILE], trainer)

tokenizer_hf.save("myth_tokenizer.json")

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="myth_tokenizer.json",
    pad_token="[PAD]",
    unk_token="[UNK]",
    bos_token="[BOS]",
    eos_token="[EOS]",
)


def encode(text):
    return hf_tokenizer.encode(text, add_special_tokens=True)

def decode(token_ids):
    return hf_tokenizer.decode(token_ids, skip_special_tokens=True)
