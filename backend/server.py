"""FastAPI server for Mistral 7B feature steering with SAE."""

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sae_lens import SAE
from transformer_lens import HookedTransformer

FEATURE_LABELS = {
    # Typing features — validated (from typing experiment)
    124809: "type annotations (cross-language) [verified]",
    6133: "static typing style [verified]",
    8019: "TypeScript type annotations",
    28468: "explicit type signatures",
    95915: "generic type parameters <T>",
    70728: "Python type hints",
}

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
SAE_ID = "tylercosgrove/mistral-7b-sparse-autoencoder-layer16"
MAX_NEW_TOKENS = 200


class FeatureOverride(BaseModel):
    id: int
    strength: float


class GenerateRequest(BaseModel):
    prompt: str
    features: list[FeatureOverride] = []
    temperature: float = 0.3


# Global model references set during startup
model: HookedTransformer | None = None
sae: SAE | None = None
hook_point: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, sae, hook_point

    print("Loading model...")
    model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, device="cuda", dtype=torch.float16)
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print("Loading SAE...")
    sae = SAE.from_pretrained(
        release=SAE_ID,
        sae_id=".",
    )[0]
    sae = sae.to("cuda")
    hook_point = sae.cfg.metadata["hook_name"]
    print(f"SAE loaded (hook: {hook_point}). VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

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
    assert model is not None and sae is not None and hook_point is not None

    temp = req.temperature
    do_sample = temp > 0

    # Baseline generation (no hooks) — generate() returns a string by default
    baseline_text = model.generate(
        req.prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=temp,
        do_sample=do_sample,
    )

    # Steered generation (with steering hook)
    active = [(f.id, f.strength) for f in req.features if f.strength != 0]

    if not active:
        return {"baseline": baseline_text, "steered": baseline_text}

    # Validate feature IDs and pre-compute detached steering vectors
    steering_vectors = []
    for feat_id, strength in active:
        if not (0 <= feat_id < sae.cfg.d_sae):
            raise ValueError(f"Feature ID {feat_id} out of range [0, {sae.cfg.d_sae})")
        steering_vectors.append((strength, sae.W_dec[feat_id].detach().clone()))

    def steering_hook(value, hook):
        if value.shape[1] == 1:
            return value
        for strength, vec in steering_vectors:
            value[:, :, :] = value + strength * vec.to(value.device, value.dtype)
        return value

    with model.hooks(fwd_hooks=[(hook_point, steering_hook)]):
        steered_text = model.generate(
            req.prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temp,
            do_sample=do_sample,
        )

    return {"baseline": baseline_text, "steered": steered_text}
