from .model import Transformer
from .data import qa_text
from .data import build_training_text
from .tokenizer import encode
import torch

#split = int(0.9 * len(text))
#train_data = text[:split]
#validation_data = text[split:]

#unshuffled data
#train_tokens = torch.tensor(encode(train_data), dtype=torch.long)
#val_tokens = torch.tensor(encode(validation_data), dtype=torch.long)
qa_tokens = torch.tensor(encode(qa_text), dtype=torch.long)


def get_batch(data):
    # pick batch_size number of indexes for beginning of blocks of length context_length
    indexes = torch.randint(len(data) - model.context_length, (model.batch_size,))
    x = torch.stack([data[i:i + model.context_length] for i in indexes])
    y = torch.stack([data[i + 1: i + model.context_length + 1] for i in indexes])

    # x, y are dimension(batch_size * context_length) and already encoded as integers
    return x, y


model = Transformer(token_size=2000, embed_size=256, batch_size=24, context_length=256, num_repetitions=4)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(2):
    #shuffled training data
    train_data = build_training_text()
    for i in range(4000):
        x, y = get_batch(train_data)

        # Fix: Squeeze the input tensor to remove the extra dimension
        x = x.squeeze(-1)

        output, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            print(f"step {i}: loss {loss.item():.4f}")

#lower learning rate for fine tuning to question answer pairs
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for i in range(1000):
    x, y = get_batch(qa_tokens)
    x = x.squeeze(-1)
    output, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"step {i}: loss {loss.item():.4f}")

torch.save(model.state_dict(), "weights.pt")
