"""FastAPI server for Ministral 8B feature steering with custom TopK SAE.

Uses HuggingFace transformers (not TransformerLens) and a custom TopKSAE
(not sae_lens).  Decode-only steering: the hook fires only during
single-token decode steps, leaving prompt prefill unmodified.

Usage:
  uvicorn backend.server:app --host 0.0.0.0 --port 8000

Environment variables:
  SAE_CHECKPOINT  — path to sae_checkpoint.pt (required)
  MODEL_PATH — path or HF ID for model (default: mistralai/Ministral-8B-Instruct-2410)
  STEER_LAYER — decoder layer index to hook (default: 18)
  MAX_NEW_TOKENS — generation length (default: 200)
"""

from __future__ import annotations

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
MODEL_PATH = os.environ.get("MODEL_PATH", "mistralai/Ministral-8B-Instruct-2410")
STEER_LAYER = int(os.environ.get("STEER_LAYER", "18"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "200"))

# ---------------------------------------------------------------------------
# Hardcoded feature registry — best single feature per property from
# discovery run (see results/demo_features.json and FEATURES_FOUND.md).
# ---------------------------------------------------------------------------

FEATURE_REGISTRY: dict[str, str] = {
    "13176": "Typing",
    "16290": "Recursion",
}

# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class FeatureOverride(BaseModel):
    id: int
    strength: float


class GenerateRequest(BaseModel):
    prompt: str
    features: list[FeatureOverride] = []
    temperature: float = 0.3


class AnalyzeRequest(BaseModel):
    prompt: str
    feature_id: int
    steering: list[FeatureOverride] = []


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

model: AutoModelForCausalLM | None = None
tokenizer: AutoTokenizer | None = None
sae: TopKSAE | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, sae

    if not SAE_CHECKPOINT:
        raise RuntimeError("SAE_CHECKPOINT environment variable not set")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load model ---
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
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

    print(f"Serving {len(FEATURE_REGISTRY)} features: {list(FEATURE_REGISTRY.keys())}")

    yield

    del model, tokenizer, sae
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
    """Return available feature labels."""
    return FEATURE_REGISTRY


@app.get("/info")
def get_info():
    """Return model and SAE metadata."""
    sae_short = Path(SAE_CHECKPOINT).name if SAE_CHECKPOINT else "unknown"
    return {"model": MODEL_PATH, "sae": sae_short, "layer": STEER_LAYER}


@app.post("/generate")
def generate(req: GenerateRequest):
    """Generate baseline and steered text."""
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
    combined_direction = torch.zeros(sae.W_dec.shape[1], device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for feat_id, strength in active:
            combined_direction += strength * sae.W_dec[feat_id].to(device=device, dtype=torch.bfloat16)

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


# ---------------------------------------------------------------------------
# Analyze endpoint — feature activation visualization
# ---------------------------------------------------------------------------


def _make_cache_hook(storage: dict, key: str):
    """Create a forward hook that captures a module's output into storage[key]."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            storage[key] = output[0].detach()
        else:
            storage[key] = output.detach()
    return hook


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Analyze feature activations on generated text.

    1. Generate text (optionally with steering).
    2. Run a clean forward pass on the full sequence, caching attention
       and MLP outputs at each layer via PyTorch hooks.
    3. Pass the residual stream at STEER_LAYER through the SAE to get
       per-token feature activations.
    4. Compute layer-by-layer attribution for the selected feature.
    """
    assert model is not None and tokenizer is not None and sae is not None

    device = next(model.parameters()).device
    d_sae = sae.W_dec.shape[0]

    # Validate feature ID
    if not (0 <= req.feature_id < d_sae):
        raise HTTPException(
            status_code=400,
            detail=f"Feature ID {req.feature_id} out of range [0, {d_sae})",
        )

    # --- Step 1: Generate text (with optional steering) ---
    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    active_steering = [(s.id, s.strength) for s in req.steering if s.strength != 0]

    if active_steering:
        # Validate steering feature IDs
        for feat_id, _ in active_steering:
            if not (0 <= feat_id < d_sae):
                raise HTTPException(
                    status_code=400,
                    detail=f"Steering feature ID {feat_id} out of range [0, {d_sae})",
                )

        combined_direction = torch.zeros(
            sae.W_dec.shape[1], device=device, dtype=torch.bfloat16
        )
        with torch.no_grad():
            for feat_id, strength in active_steering:
                combined_direction += strength * sae.W_dec[feat_id].to(
                    device=device, dtype=torch.bfloat16
                )

        hook_fn = make_steering_hook(combined_direction, alpha=1.0)
        steer_handle = model.model.layers[STEER_LAYER].register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                full_ids = model.generate(**inputs, **gen_kwargs)
        finally:
            steer_handle.remove()
    else:
        with torch.no_grad():
            full_ids = model.generate(**inputs, **gen_kwargs)

    # --- Step 2: Forward pass with caching hooks (no steering) ---
    cache: dict[str, torch.Tensor] = {}
    handles: list = []

    num_layers = STEER_LAYER + 1  # layers 0 through STEER_LAYER

    for i in range(num_layers):
        h = model.model.layers[i].self_attn.register_forward_hook(
            _make_cache_hook(cache, f"attn_{i}")
        )
        handles.append(h)
        h = model.model.layers[i].mlp.register_forward_hook(
            _make_cache_hook(cache, f"mlp_{i}")
        )
        handles.append(h)

    # Also cache the residual stream after STEER_LAYER
    h = model.model.layers[STEER_LAYER].register_forward_hook(
        _make_cache_hook(cache, "resid_post")
    )
    handles.append(h)

    try:
        with torch.no_grad():
            model(full_ids)
    finally:
        for h in handles:
            h.remove()

    # --- Step 3: SAE feature activations ---
    resid = cache["resid_post"]  # (1, seq_len, d_model)
    resid_2d = resid[0]  # (seq_len, d_model)
    seq_len = resid_2d.shape[0]

    # SAE forward applies TopK sparsity
    with torch.no_grad():
        x_hat, topk_latents, _ = sae(resid_2d.float())
        # topk_latents: (seq_len, d_sae) — sparse, only top-k nonzero per row

    # Heatmap data: selected feature's activation per token
    feature_acts = topk_latents[:, req.feature_id].cpu().tolist()

    # Feature direction for layer attribution
    feat_direction = sae.W_dec[req.feature_id].float()  # (d_model,)

    # Decode tokens
    tokens = [tokenizer.decode(t) for t in full_ids[0]]

    # Feature label
    feat_label = FEATURE_REGISTRY.get(
        str(req.feature_id), f"Feature {req.feature_id}"
    )

    # --- Step 4: Per-token details ---
    token_details: dict[str, dict] = {}

    for pos in range(seq_len):
        # SAE decomposition: top active features at this position
        pos_acts = topk_latents[pos]  # (d_sae,)
        nonzero_mask = pos_acts != 0
        nonzero_indices = nonzero_mask.nonzero(as_tuple=False).squeeze(-1)

        if nonzero_indices.numel() > 0:
            nonzero_values = pos_acts[nonzero_indices]
            sorted_order = nonzero_values.argsort(descending=True)
            top_indices = nonzero_indices[sorted_order][:10]
            top_values = nonzero_values[sorted_order][:10]

            sae_decomp = [
                {
                    "feature_id": idx.item(),
                    "label": FEATURE_REGISTRY.get(
                        str(idx.item()), f"Feature {idx.item()}"
                    ),
                    "activation": round(val.item(), 4),
                }
                for idx, val in zip(top_indices, top_values)
            ]
        else:
            sae_decomp = []

        # Layer attribution: dot product of each layer's attn/MLP output
        # with the feature direction
        layer_attrib = []
        for layer_idx in range(num_layers):
            attn_out = cache[f"attn_{layer_idx}"]  # (1, seq_len, d_model)
            mlp_out = cache[f"mlp_{layer_idx}"]

            attn_proj = (attn_out[0, pos].float() @ feat_direction).item()
            mlp_proj = (mlp_out[0, pos].float() @ feat_direction).item()

            layer_attrib.append({
                "layer": layer_idx,
                "attn": round(attn_proj, 4),
                "mlp": round(mlp_proj, 4),
            })

        # Reconstruction error at this position
        recon_error = (resid_2d[pos].float() - x_hat[pos]).pow(2).sum().sqrt().item()

        token_details[str(pos)] = {
            "sae_decomposition": sae_decomp,
            "layer_attribution": layer_attrib,
            "reconstruction_error": round(recon_error, 4),
        }

    return {
        "tokens": tokens,
        "feature_label": feat_label,
        "feature_activations": feature_acts,
        "token_details": token_details,
    }
