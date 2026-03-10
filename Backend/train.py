from model import Transformer
from data import qa_text
from data import build_training_text
from tokenizer import encode
import torch


qa_tokens = torch.tensor(encode(qa_text), dtype=torch.long)

split = int(0.9 * len(qa_tokens))

train_tokens = qa_tokens[:split]
val_tokens = qa_tokens[split:]


def get_batch(data):
    # pick batch_size number of indexes for beginning of blocks of length context_length
    indexes = torch.randint(len(data) - model.context_length, (model.batch_size,))
    x = torch.stack([data[i:i + model.context_length] for i in indexes])
    y = torch.stack([data[i + 1: i + model.context_length + 1] for i in indexes])

    # x, y are dimension(batch_size * context_length) and already encoded as integers
    return x, y


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)
model = Transformer(token_size=2000, embed_size=256, batch_size=24, context_length=256, num_repetitions=4).to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(2):

    train_data = build_training_text()

    for i in range(4000):

        model.train()

        x, y = get_batch(train_tokens)

        x = x.squeeze(-1).to(device)
        y = y.to(device)

        output, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 500 == 0:

            model.eval()

            with torch.no_grad():
                x_val, y_val = get_batch(val_tokens)
                x_val = x_val.squeeze(-1).to(device)
                y_val = y_val.to(device)

                _, val_loss = model(x_val, y_val)

            print(
                f"epoch {epoch} step {i} | train loss {loss.item():.4f} | val loss {val_loss.item():.4f}"
            )

