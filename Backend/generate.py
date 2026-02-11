import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, idx, max_new_tokens, context_length, temperature=1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
    return idx
