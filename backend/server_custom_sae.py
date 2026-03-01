"""FastAPI server for feature steering with our custom-trained TopK SAE.

Same API contract as server.py (GET /features, POST /generate) so the frontend
needs zero changes. Key differences from the original server:
  - HuggingFace model (not TransformerLens)
  - Custom TopKSAE (not sae_lens)
  - Decode-only steering (not prefill) — per plan.md rationale
  - Rich feature labels from pipeline analysis (monotonicity, density, effect size)

Usage:
  uvicorn backend.server_custom_sae:app --host 0.0.0.0 --port 8000

Environment variables:
  SAE_CHECKPOINT       — path to sae_checkpoint.pt (required)
  FEATURE_CANDIDATES   — path to feature_candidates.json (required)
  STEERING_RESULTS     — path to steering_results.json (optional, enriches labels)
  MODEL_PATH           — path or HF ID for model (default: mistralai/Mistral-7B-Instruct-v0.3)
  STEER_LAYER          — decoder layer index to hook (default: 16)
  MAX_NEW_TOKENS       — generation length (default: 200)
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
STEERING_RESULTS = os.environ.get("STEERING_RESULTS", "")
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


def _build_feature_label(candidate: dict, steering_analysis: dict) -> str:
    """Build a rich human-readable label for a feature using pipeline analysis.

    Uses monotonicity and effect size from steering_results.json when available,
    falling back to basic Cohen's d label otherwise.

    Examples:
        "control_flow (monotonic, effect=+0.179, d=0.45)"
        "error_handling (d=0.32)"
    """
    idx = str(candidate["feature_idx"])
    variant = candidate.get("primary_variant", "unknown")
    cohens_d = candidate.get("cohens_d", 0)

    # Try to find monotonicity data for this feature's steering direction
    # Steering directions are keyed by feature_idx in steering_results
    mono_info = _find_monotonicity(idx, steering_analysis)
    if mono_info:
        prop_name, effect, is_mono = mono_info
        tag = "monotonic" if is_mono else "non-monotonic"
        # If the steered property differs from the selected variant, show both
        if prop_name != variant:
            return f"{variant} -> {prop_name} ({tag}, effect={effect:+.3f}, d={cohens_d:.2f})"
        return f"{prop_name} ({tag}, effect={effect:+.3f}, d={cohens_d:.2f})"

    return f"{variant} (d={cohens_d:.2f})"


def _find_monotonicity(
    direction_name: str, steering_analysis: dict
) -> tuple[str, float, bool] | None:
    """Find the strongest monotonicity signal for a direction across experiments.

    Returns (property_name, effect_size, is_monotonic) for the property with
    the largest |effect_size|, or None if no data found.
    """
    for exp_data in steering_analysis.values():
        mono = exp_data.get("monotonicity", {})
        if direction_name not in mono:
            continue
        props = mono[direction_name]
        if not props:
            continue
        best_prop, best_data = max(props.items(), key=lambda x: abs(x[1]["effect_size"]))
        return (best_prop, best_data["effect_size"], best_data["is_monotonic"])
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, sae, feature_labels

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

    # --- Load feature labels (enriched with steering analysis if available) ---
    print(f"Loading feature candidates from {FEATURE_CANDIDATES}...")
    with open(FEATURE_CANDIDATES) as f:
        candidates_data = json.load(f)

    # Load steering analysis for monotonicity/density enrichment
    steering_analysis = {}
    if STEERING_RESULTS and Path(STEERING_RESULTS).exists():
        print(f"Loading steering analysis from {STEERING_RESULTS}...")
        with open(STEERING_RESULTS) as f:
            steering_analysis = json.load(f)

    feature_labels.clear()
    for c in candidates_data["candidates"]:
        idx = c["feature_idx"]
        feature_labels[str(idx)] = _build_feature_label(c, steering_analysis)

    for idx in candidates_data.get("random_control_features", []):
        feature_labels[str(idx)] = f"random control #{idx}"

    print(f"Serving {len(feature_labels)} features:")
    for fid, label in feature_labels.items():
        print(f"  {fid}: {label}")

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


@app.get("/info")
def get_info():
    """Return server configuration (model, layer, SAE size)."""
    return {
        "model": MODEL_PATH,
        "steer_layer": STEER_LAYER,
        "d_sae": sae.d_sae if sae else 0,
        "max_new_tokens": MAX_NEW_TOKENS,
    }


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
