"""FastAPI server for Mistral 7B feature steering with our custom-trained TopK SAE.

Same API contract as server.py (GET /features, POST /generate) so the frontend
needs zero changes. Key differences from the original server:
  - HuggingFace model (not TransformerLens)
  - Custom TopKSAE (not sae_lens)
  - Decode-only steering (not prefill) — per plan.md rationale
  - Mistral v0.3 (not v0.1)

Usage:
  uvicorn backend.server_custom_sae:app --host 0.0.0.0 --port 8000

Environment variables:
  SAE_CHECKPOINT  — path to sae_checkpoint.pt (required)
  FEATURE_CANDIDATES — path to feature_candidates.json (required)
  MODEL_PATH — path or HF ID for model (default: mistralai/Mistral-7B-Instruct-v0.3)
  STEER_LAYER — decoder layer index to hook (default: 16)
  MAX_NEW_TOKENS — generation length (default: 200)
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
# Add project root so experiments package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.sae.model import TopKSAE
from experiments.steering.hook import make_steering_hook

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

SAE_CHECKPOINT = os.environ.get("SAE_CHECKPOINT", "")
FEATURE_CANDIDATES = os.environ.get("FEATURE_CANDIDATES", "")
MODEL_PATH = os.environ.get("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.3")
STEER_LAYER = int(os.environ.get("STEER_LAYER", "16"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "200"))

# ---------------------------------------------------------------------------
# Request/response models (identical to server.py)
# ---------------------------------------------------------------------------


class FeatureOverride(BaseModel):
    id: int
    strength: float


class GenerateRequest(BaseModel):
    prompt: str
    features: list[FeatureOverride] = []
    temperature: float = 0.3


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

model: AutoModelForCausalLM | None = None
tokenizer: AutoTokenizer | None = None
sae: TopKSAE | None = None
feature_labels: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, sae, feature_labels, decoder_directions

    if not SAE_CHECKPOINT:
        raise RuntimeError("SAE_CHECKPOINT environment variable not set")
    if not FEATURE_CANDIDATES:
        raise RuntimeError("FEATURE_CANDIDATES environment variable not set")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load model ---
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # --- Load SAE ---
    print(f"Loading SAE from {SAE_CHECKPOINT}...")
    ckpt = torch.load(SAE_CHECKPOINT, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sae = TopKSAE(d_model=cfg["d_model"], d_sae=cfg["d_sae"], k=cfg["k"])
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(device)
    sae.eval()
    print(f"SAE loaded: {cfg['d_sae']} features, k={cfg['k']}")

    # --- Load feature labels ---
    print(f"Loading feature candidates from {FEATURE_CANDIDATES}...")
    with open(FEATURE_CANDIDATES) as f:
        candidates_data = json.load(f)

    for c in candidates_data["candidates"]:
        idx = c["feature_idx"]
        variant = c.get("primary_variant", "unknown")
        d = c.get("cohens_d", 0)
        feature_labels[str(idx)] = f"{variant} feature (d={d:.2f})"

    # Also expose random controls as labeled features
    for idx in candidates_data.get("random_control_features", []):
        feature_labels[str(idx)] = f"random control #{idx}"

    print(f"Serving {len(feature_labels)} features: {list(feature_labels.keys())}")

    yield

    del model, tokenizer, sae
    torch.cuda.empty_cache()


app = FastAPI(title="Custom SAE Feature Steering API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/features")
def get_features():
    """Return available feature labels. Same contract as server.py."""
    return feature_labels


@app.post("/generate")
def generate(req: GenerateRequest):
    """Generate baseline and steered text. Same contract as server.py."""
    assert model is not None and tokenizer is not None and sae is not None

    device = next(model.parameters()).device
    temp = req.temperature
    do_sample = temp > 0

    # Tokenize prompt
    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=temp if do_sample else None,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )

    # --- Baseline generation (no hooks) ---
    with torch.no_grad():
        baseline_ids = model.generate(**inputs, **gen_kwargs)
    baseline_text = tokenizer.decode(baseline_ids[0][prompt_len:], skip_special_tokens=True)

    # --- Steered generation ---
    active = [(f.id, f.strength) for f in req.features if f.strength != 0]

    if not active:
        return {"baseline": baseline_text, "steered": baseline_text}

    # Validate feature IDs
    d_sae = sae.W_dec.shape[0]
    for feat_id, _ in active:
        if not (0 <= feat_id < d_sae):
            raise HTTPException(
                status_code=400,
                detail=f"Feature ID {feat_id} out of range [0, {d_sae})",
            )

    # Compute combined steering direction (sum of all active feature directions)
    combined_direction = torch.zeros(sae.W_dec.shape[1], device=device, dtype=torch.float16)
    with torch.no_grad():
        for feat_id, strength in active:
            combined_direction += strength * sae.W_dec[feat_id].to(device=device, dtype=torch.float16)

    # Attach decode-only steering hook at the configured layer
    hook_fn = make_steering_hook(combined_direction, alpha=1.0)  # strength already in direction
    handle = model.model.layers[STEER_LAYER].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            steered_ids = model.generate(**inputs, **gen_kwargs)
        steered_text = tokenizer.decode(steered_ids[0][prompt_len:], skip_special_tokens=True)
    finally:
        handle.remove()

    return {"baseline": baseline_text, "steered": steered_text}
