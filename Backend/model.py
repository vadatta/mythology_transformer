import torch.nn as nn
import torch
import torch.nn.functional as F

class Embedding(nn.Module):
  def __init__(self, token_size, embed_size, context_length):
    super().__init__()
    self.token_embeddings = nn.Embedding(token_size, embed_size)
    self.positional_embeddings = nn.Embedding(context_length, embed_size)


  def forward(self, x):
    b, t = x.shape
    token_embed = self.token_embeddings(x)
    positional_embed = self.positional_embeddings(
      torch.arange(t, device=x.device)
    )

    # (batch_size * context_length * embed_size)
    return token_embed + positional_embed


class MultiHeadAttention(nn.Module):
  def __init__(self, num_embeddings):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.key = nn.Linear(num_embeddings, num_embeddings, bias = False)
    self.query = nn.Linear(num_embeddings, num_embeddings, bias = False)
    self.values = nn.Linear(num_embeddings, num_embeddings, bias = False)

  def forward(self, x):
    b, t, c = x.shape
    #x is batch+size * context_length * embeddding_size
    key = self.key(x) #batch+size * context_length * embeddding_size
    query = self.query(x) #batch+size * context_length * embeddding_size
    values = self.values(x) #batch+size * context_length * embeddding_size
    weights = (query @ torch.transpose(key, -1, -2)) / (self.num_embeddings ** 0.5)

    #prior to softmax, we must max values to prevent learning from future tokens

    #creates a lower triangular matrix of all ones
    mask = torch.tril(torch.ones(t, t, device=x.device))

    #upper triangular part of weights is masked to -infinity to prevent tokens
    #at pos t to learn from tokens as pot > t
    weights = weights.masked_fill(mask == 0, float('-inf'))

    output = F.softmax(weights, dim = -1) @ values

    return output

class FeedForwardLayer(nn.Module):
  def __init__(self, num_embeddings):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.net = nn.Sequential(
        nn.Linear(num_embeddings, num_embeddings * 4),
        nn.ReLU(),
        nn.Linear(num_embeddings * 4, num_embeddings))

  def forward(self, x):
    return self.net(x)


class TransformerBlock(nn.Module):
  def __init__(self, num_embeddings):
    super().__init__()

    self.layer1 = nn.LayerNorm(num_embeddings)
    self.attention = MultiHeadAttention(num_embeddings)

    self.layer2 = nn.LayerNorm(num_embeddings)
    self.ffl = FeedForwardLayer(num_embeddings)

  def forward(self, x):
    #residual
    x = x + self.attention(self.layer1(x))
    x = x + self.ffl(self.layer2(x))
    return x


class Transformer(nn.Module):
  def __init__(self, token_size, embed_size, batch_size, context_length, num_repetitions):
    super().__init__()
    self.batch_size = batch_size
    self.context_length = context_length
    self.embeddings = Embedding(token_size, embed_size, context_length)
    self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_size) for i in range(num_repetitions)])

    self.fl = nn.LayerNorm(embed_size)
    self.fn = nn.Linear(embed_size, token_size)


  def forward(self, indexes, targets = None):
    x = self.embeddings(indexes)
    for block in self.transformer_blocks:
      x = block(x)

    x = self.fl(x)
    output = self.fn(x)

    if targets is None:
        return output
    else:
        B, T, C = output.shape
        loss = F.cross_entropy(
                output.view(B*T, C),
                targets.view(B*T)
            )
        return output, loss


