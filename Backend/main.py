from fastapi import FastAPI
from pydantic import BaseModel
import torch

from .model import Transformer
from .generate import generate
from .tokenizer import encode, decode

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, OPTIONS, etc.
    allow_headers=["*"],
)


# ---- load model once ----
device = "cpu"

model = Transformer(
    token_size=2000, embed_size=256, batch_size=24, context_length=256, num_repetitions=4
)
model.load_state_dict(torch.load("weights.pt", map_location=device))
model.eval()

# ---- request schema ----
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.8

# ---- endpoint ----
@app.post("/generate")
def generate_text(req: GenerateRequest):
    idx = torch.tensor([encode(req.prompt)], dtype=torch.long)

    with torch.no_grad():
        out = generate(
            model=model,
            idx=idx,
            max_new_tokens=req.max_new_tokens,
            context_length=16,
            temperature=req.temperature
        )

    text = decode(out[0].tolist())
    return {"completion": text}

