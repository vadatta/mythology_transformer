from model import Transformer
from data import train_set, val_set, collate
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=collate)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)
model = Transformer(token_size=2000, embed_size=256, batch_size=8, context_length=256, num_repetitions=4).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(20):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)  # (B, T)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()

        logits = model(input_ids)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, C), labels.reshape(B * T), ignore_index=-100)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)  # (B, T)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, C), labels.reshape(B * T), ignore_index=-100)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}")
