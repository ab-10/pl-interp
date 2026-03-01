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
  FEATURE_MAP          — path to feature_map.json (optional, enables /feature_map endpoint)
"""

from __future__ import annotations

import json
import os
import re
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
# Inline density computation to avoid deep import chain from analyze_steering
_DENSITY_PATTERNS: dict[str, list[str]] = {
    "type_annotations": [
        r":\s*(?:int|float|str|bool|list|dict|tuple|set|None|Any|Optional|Union|Callable)\b",
        r"->\s*(?:int|float|str|bool|list|dict|tuple|set|None|Any|Optional|Union)\b",
        r":\s*(?:List|Dict|Tuple|Set)\[",
    ],
    "error_handling": [r"\btry\b", r"\bexcept\b", r"\bcatch\b", r"\bfinally\b", r"\braise\b", r"\bthrow\b"],
    "control_flow": [r"\bif\b", r"\belif\b", r"\belse\b", r"\bfor\b", r"\bwhile\b", r"\bbreak\b", r"\bcontinue\b", r"\bswitch\b", r"\bmatch\b"],
    "decomposition": [r"\bdef\s+\w+", r"\bclass\s+\w+", r"\bfunction\s+\w+", r"\bimport\b", r"\bfrom\s+\w+\s+import\b"],
    "functional_style": [r"\bmap\s*\(", r"\bfilter\s*\(", r"\breduce\s*\(", r"\blambda\b", r"\[.+\bfor\b.+\bin\b.+\]"],
    "recursion": [r"\breturn\s+\w+\s*\(", r"\brecurs"],
    "verbose_documentation": [r'"""', r"'''", r"#\s+\S", r"//\s+\S", r"/\*", r"\bArgs:\b", r"\bReturns:\b"],
}

def compute_all_densities(text: str) -> dict[str, float]:
    """Compute density for every known property."""
    result = {}
    lines = text.strip().split("\n")
    total_lines = max(len(lines), 1)
    for prop, patterns in _DENSITY_PATTERNS.items():
        total_matches = sum(len(re.findall(p, text, re.MULTILINE | re.IGNORECASE)) for p in patterns)
        result[prop] = total_matches / total_lines
    return result

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

SAE_CHECKPOINT = os.environ.get("SAE_CHECKPOINT", "")
FEATURE_CANDIDATES = os.environ.get("FEATURE_CANDIDATES", "")
STEERING_RESULTS = os.environ.get("STEERING_RESULTS", "")
MODEL_PATH = os.environ.get("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.3")
STEER_LAYER = int(os.environ.get("STEER_LAYER", "16"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "200"))
FEATURE_MAP = os.environ.get("FEATURE_MAP", "")

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
    include_activations: bool = False
    alphas: list[float] | None = None


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

model: AutoModelForCausalLM | None = None
tokenizer: AutoTokenizer | None = None
sae: TopKSAE | None = None
feature_registry: dict[str, dict] = {}
feature_map_data: dict | None = None


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


def _get_monotonicity_data(direction_name: str, steering_analysis: dict) -> dict | None:
    """Return the full monotonicity dict for all properties for a direction, or None."""
    for exp_data in steering_analysis.values():
        mono = exp_data.get("monotonicity", {})
        if direction_name in mono and mono[direction_name]:
            return mono[direction_name]
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, sae, feature_registry, feature_map_data

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

    feature_registry.clear()
    for c in candidates_data["candidates"]:
        idx = c["feature_idx"]
        feature_registry[str(idx)] = {
            "label": _build_feature_label(c, steering_analysis),
            "primary_variant": c.get("primary_variant"),
            "cohens_d": c.get("cohens_d"),
            "category": "steering",
            "slider": {"min": -5, "max": 5, "step": 0.5, "default": 0},
            "monotonicity": _get_monotonicity_data(str(idx), steering_analysis),
        }

    for idx in candidates_data.get("random_control_features", []):
        feature_registry[str(idx)] = {
            "label": f"random control #{idx}",
            "category": "control",
            "slider": {"min": -5, "max": 5, "step": 0.5, "default": 0},
        }

    # --- Pre-compute logit attribution for each feature ---
    print("Computing logit attribution for features...")
    lm_head_weight = model.lm_head.weight  # shape (vocab_size, d_model)
    with torch.no_grad():
        for fid_str in feature_registry:
            feat_id = int(fid_str)
            direction = sae.W_dec[feat_id].to(device=lm_head_weight.device, dtype=lm_head_weight.dtype)
            logit_effect = direction @ lm_head_weight.T  # shape (vocab_size,)

            # Top-10 promoted (highest logit effect)
            top_vals, top_ids = torch.topk(logit_effect, k=10)
            promoted_tokens = tokenizer.convert_ids_to_tokens(top_ids.tolist())
            promoted = []
            for tok, val in zip(promoted_tokens, top_vals.tolist()):
                # Skip special tokens (starting with < or common padding)
                if tok.startswith("<") or tok.strip() == "":
                    continue
                promoted.append({"token": tok, "logit": round(val, 4)})

            # Top-10 suppressed (lowest logit effect)
            bot_vals, bot_ids = torch.topk(logit_effect, k=10, largest=False)
            suppressed_tokens = tokenizer.convert_ids_to_tokens(bot_ids.tolist())
            suppressed = []
            for tok, val in zip(suppressed_tokens, bot_vals.tolist()):
                if tok.startswith("<") or tok.strip() == "":
                    continue
                suppressed.append({"token": tok, "logit": round(val, 4)})

            feature_registry[fid_str]["logit_attribution"] = {
                "promoted": promoted[:10],
                "suppressed": suppressed[:10],
            }
    print("Logit attribution computed.")

    print(f"Serving {len(feature_registry)} features:")
    for fid, entry in feature_registry.items():
        print(f"  {fid}: {entry['label']}")

    # --- Load feature map if configured ---
    if FEATURE_MAP and Path(FEATURE_MAP).exists():
        print(f"Loading feature map from {FEATURE_MAP}...")
        with open(FEATURE_MAP) as f:
            feature_map_data = json.load(f)
        print(f"Feature map loaded.")

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
    """Return available feature registry with enriched metadata."""
    return feature_registry


@app.get("/info")
def get_info():
    """Return server configuration (model, layer, SAE size, capabilities)."""
    return {
        "model": MODEL_PATH,
        "steer_layer": STEER_LAYER,
        "d_sae": sae.d_sae if sae else 0,
        "max_new_tokens": MAX_NEW_TOKENS,
        "capabilities": {
            "token_activations": True,
            "alpha_sweep": True,
            "feature_map": bool(FEATURE_MAP),
            "enriched_features": True,
            "density": True,
        },
    }


@app.get("/feature_map")
def get_feature_map():
    """Return the feature map data, if available."""
    if feature_map_data is None:
        raise HTTPException(status_code=404, detail="Feature map not available")
    return feature_map_data


def _compute_activation_stats(
    token_activations_list: list[dict], active: list[tuple[int, float]]
) -> dict:
    """Compute per-feature activation summary statistics across all tokens."""
    stats = {}
    for feat_id, _ in active:
        feat_id_str = str(feat_id)
        values = []
        for tok in token_activations_list:
            v = tok["activations"].get(feat_id_str, 0)
            if v != 0:
                values.append(v)
        if values:
            stats[feat_id_str] = {
                "count": len(values),
                "total_tokens": len(token_activations_list),
                "sparsity": 1 - len(values) / len(token_activations_list),
                "mean": sum(values) / len(values),
                "max": max(values),
                "min": min(values),
            }
    return stats


def _compute_top_activating_tokens(
    token_activations_list: list[dict], active: list[tuple[int, float]]
) -> dict:
    """Find the top-10 tokens with highest activation for each active feature."""
    top_tokens = {}
    for feat_id, _ in active:
        feat_id_str = str(feat_id)
        token_acts = []
        for tok in token_activations_list:
            v = tok["activations"].get(feat_id_str, 0)
            if v != 0:
                token_acts.append({"token": tok["token"], "activation": v})
        token_acts.sort(key=lambda x: x["activation"], reverse=True)
        top_tokens[feat_id_str] = token_acts[:10]
    return top_tokens


@app.post("/generate")
def generate(req: GenerateRequest):
    """Generate baseline and steered text with optional activation capture, density, and alpha sweep."""
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
        response = {"baseline": baseline_text, "steered": baseline_text}
        response["baseline_density"] = compute_all_densities(baseline_text)
        response["steered_density"] = compute_all_densities(baseline_text)
        return response

    # Validate feature IDs
    d_sae = sae.W_dec.shape[0]
    for feat_id, _ in active:
        if not (0 <= feat_id < d_sae):
            raise HTTPException(
                status_code=400,
                detail=f"Feature ID {feat_id} out of range [0, {d_sae})",
            )

    active_feat_ids = set(feat_id for feat_id, _ in active)

    # Compute combined steering direction (sum of all active feature directions)
    combined_direction = torch.zeros(sae.W_dec.shape[1], device=device, dtype=torch.float16)
    with torch.no_grad():
        for feat_id, strength in active:
            combined_direction += strength * sae.W_dec[feat_id].to(device=device, dtype=torch.float16)

    # --- Helper: generate with a given direction and optional activation capture ---
    def _generate_with_direction(direction: torch.Tensor, alpha: float, capture_activations: bool):
        """Generate text with a steering hook. Returns (text, token_activations_or_None)."""
        hook_fn = make_steering_hook(direction, alpha=alpha)
        handle = model.model.layers[STEER_LAYER].register_forward_hook(hook_fn)

        captured_hidden_states: list[torch.Tensor] = []
        capture_handle = None

        if capture_activations and active_feat_ids:
            def capture_hook(module, args, output):
                is_tuple = isinstance(output, tuple)
                hidden_states = output[0] if is_tuple else output
                if hidden_states.shape[1] == 1:
                    # Decode step: capture hidden states (detach + clone to avoid memory issues)
                    captured_hidden_states.append(hidden_states.detach().clone().squeeze(1))
                return output

            capture_handle = model.model.layers[STEER_LAYER].register_forward_hook(capture_hook)

        try:
            with torch.no_grad():
                gen_ids = model.generate(**inputs, **gen_kwargs)
            text = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True)
        finally:
            handle.remove()
            if capture_handle is not None:
                capture_handle.remove()

        # Build token activations if captured
        token_activations = None
        if capture_activations and active_feat_ids and captured_hidden_states:
            stacked = torch.cat(captured_hidden_states, dim=0)  # (n_tokens, d_model)
            with torch.no_grad():
                _x_hat, _topk_latents, sae_info = sae.forward(stacked)
            topk_indices = sae_info["topk_indices"]  # (n_tokens, k)
            topk_values = sae_info["topk_values"]    # (n_tokens, k)

            # Get generated token IDs (excluding prompt)
            generated_token_ids = gen_ids[0][prompt_len:].tolist()
            token_strings = tokenizer.convert_ids_to_tokens(generated_token_ids)

            n_tokens = min(len(token_strings), topk_indices.shape[0])
            token_activations = []
            for pos in range(n_tokens):
                pos_indices = topk_indices[pos].tolist()
                pos_values = topk_values[pos].tolist()
                # Check which active features fired at this position
                active_at_pos = {}
                for i, feat_idx in enumerate(pos_indices):
                    if feat_idx in active_feat_ids:
                        active_at_pos[str(feat_idx)] = pos_values[i]
                token_activations.append({
                    "token": token_strings[pos],
                    "activations": active_at_pos,
                })

        return text, token_activations

    # --- Main steered generation (alpha=1.0, strength already baked into direction) ---
    steered_text, steered_token_activations = _generate_with_direction(
        combined_direction, alpha=1.0, capture_activations=req.include_activations
    )

    response = {"baseline": baseline_text, "steered": steered_text}

    if req.include_activations and steered_token_activations is not None:
        response["token_activations"] = steered_token_activations

        # --- Activation stats per feature ---
        response["activation_stats"] = _compute_activation_stats(steered_token_activations, active)

        # --- Top activating tokens per feature ---
        response["top_activating_tokens"] = _compute_top_activating_tokens(steered_token_activations, active)

    # --- Density computation ---
    response["baseline_density"] = compute_all_densities(baseline_text)
    response["steered_density"] = compute_all_densities(steered_text)

    # --- Alpha sweep ---
    if req.alphas is not None:
        # Compute a base direction WITHOUT strength multiplication
        # (just the sum of W_dec[feat_id] for each active feature, weighted by strength but NOT by alpha)
        # The combined_direction already has strength baked in, so we use it as the base direction.
        # Each alpha in the sweep multiplies this base direction.
        sweep_results = []
        for alpha_val in req.alphas:
            sweep_text, sweep_token_activations = _generate_with_direction(
                combined_direction, alpha=alpha_val, capture_activations=req.include_activations
            )
            sweep_entry = {"alpha": alpha_val, "text": sweep_text}
            if req.include_activations and sweep_token_activations is not None:
                sweep_entry["token_activations"] = sweep_token_activations
                sweep_entry["activation_stats"] = _compute_activation_stats(sweep_token_activations, active)
                sweep_entry["top_activating_tokens"] = _compute_top_activating_tokens(sweep_token_activations, active)
            sweep_results.append(sweep_entry)
        response["sweep_results"] = sweep_results

    return response
