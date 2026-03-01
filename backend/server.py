"""FastAPI server for Mistral 7B feature steering with SAE."""

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sae_lens import SAE
from transformer_lens import HookedTransformer

FEATURE_LABELS = {
    12057: "recursion / recursive calls",
    7291: "list comprehension / generator expressions",
    3894: "error handling / try-except blocks",
    9410: "type annotations / type hints",
    15832: "object-oriented / class definitions",
}

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
SAE_ID = "tylercosgrove/mistral-7b-sparse-autoencoder-layer16"
HOOK_POINT = "blocks.16.hook_resid_post"
MAX_NEW_TOKENS = 200


class FeatureOverride(BaseModel):
    id: int
    strength: float


class GenerateRequest(BaseModel):
    prompt: str
    features: list[FeatureOverride] = []


# Global model references set during startup
model: HookedTransformer | None = None
sae: SAE | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, sae

    print("Loading model...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda")
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print("Loading SAE...")
    sae = SAE.from_pretrained(
        release=SAE_ID,
        sae_id=".",
    )[0]
    sae = sae.to("cuda")
    print(f"SAE loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    yield

    del model, sae
    torch.cuda.empty_cache()


app = FastAPI(title="Feature Steering API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/features")
def get_features():
    return {str(k): v for k, v in FEATURE_LABELS.items()}


@app.post("/generate")
def generate(req: GenerateRequest):
    assert model is not None and sae is not None

    # Baseline generation (no hooks) — generate() returns a string by default
    baseline_text = model.generate(
        req.prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0,
    )

    # Steered generation (with steering hook)
    active = [(f.id, f.strength) for f in req.features if f.strength != 0]

    if not active:
        return {"baseline": baseline_text, "steered": baseline_text}

    def steering_hook(value, hook):
        for feat_id, strength in active:
            value = value + strength * sae.W_dec[feat_id]
        return value

    with model.hooks(fwd_hooks=[(HOOK_POINT, steering_hook)]):
        steered_text = model.generate(
            req.prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0,
        )

    return {"baseline": baseline_text, "steered": steered_text}
