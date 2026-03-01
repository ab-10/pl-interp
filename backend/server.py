"""FastAPI server for Ministral 8B feature steering with dual-layer SAEs."""

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sae_lens import SAE
from transformer_lens import HookedTransformer

MODEL_NAME = "mistralai/Ministral-8B-Instruct-2410"
SAE_LAYERS = [18, 27]
SAE_PATHS = {
    18: "/home/azureuser/8b_saes/layer18",
    27: "/home/azureuser/8b_saes/layer27",
}
MAX_NEW_TOKENS = 200

# Per-layer feature labels — populated after feature discovery
FEATURE_LABELS: dict[int, dict[int, str]] = {
    18: {},
    27: {},
}


class FeatureOverride(BaseModel):
    id: int
    layer: int
    strength: float


class GenerateRequest(BaseModel):
    prompt: str
    features: list[FeatureOverride] = []
    temperature: float = 0.3


# Global model references set during startup
model: HookedTransformer | None = None
saes: dict[int, SAE] = {}
hook_points: dict[int, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, saes, hook_points

    print("Loading model...")
    model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, device="cuda", dtype=torch.float16)
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    for layer in SAE_LAYERS:
        path = SAE_PATHS[layer]
        print(f"Loading SAE layer {layer} from {path}...")
        sae = SAE.load_from_disk(path, device="cuda")
        hp = f"blocks.{layer}.hook_resid_post"
        saes[layer] = sae
        hook_points[layer] = hp
        print(f"  SAE layer {layer} loaded (hook: {hp}). VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    yield

    del model
    saes.clear()
    hook_points.clear()
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
    return {
        str(layer): {str(k): v for k, v in labels.items()}
        for layer, labels in FEATURE_LABELS.items()
    }


@app.get("/info")
def get_info():
    sae_shorts = {}
    for layer, path in SAE_PATHS.items():
        sae_shorts[str(layer)] = "/".join(path.rstrip("/").split("/")[-2:])
    return {"model": MODEL_NAME, "saes": sae_shorts}


@app.post("/generate")
def generate(req: GenerateRequest):
    assert model is not None and saes

    temp = req.temperature
    do_sample = temp > 0

    # Baseline generation (no hooks) — generate() returns a string by default
    baseline_text = model.generate(
        req.prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=temp,
        do_sample=do_sample,
    )

    # Steered generation (with steering hooks)
    active = [(f.layer, f.id, f.strength) for f in req.features if f.strength != 0]

    if not active:
        return {"baseline": baseline_text, "steered": baseline_text}

    # Group by layer, validate feature IDs, pre-compute steering vectors
    hooks = []
    for layer in SAE_LAYERS:
        layer_feats = [(fid, s) for l, fid, s in active if l == layer]
        if not layer_feats:
            continue

        sae = saes.get(layer)
        hp = hook_points.get(layer)
        if sae is None or hp is None:
            raise ValueError(f"SAE for layer {layer} not loaded")

        steering_vectors = []
        for feat_id, strength in layer_feats:
            if not (0 <= feat_id < sae.cfg.d_sae):
                raise ValueError(f"Feature ID {feat_id} out of range [0, {sae.cfg.d_sae}) for layer {layer}")
            steering_vectors.append((strength, sae.W_dec[feat_id].detach().clone()))

        def make_hook(vecs):
            seen_prompt = False

            def steering_hook(value, hook):
                nonlocal seen_prompt
                if not seen_prompt:
                    seen_prompt = True
                    return value  # skip prompt encoding
                for strength, vec in vecs:
                    value[:, :, :] = value + strength * vec.to(value.device, value.dtype)
                return value
            return steering_hook

        hooks.append((hp, make_hook(steering_vectors)))

    with model.hooks(fwd_hooks=hooks):
        steered_text = model.generate(
            req.prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temp,
            do_sample=do_sample,
        )

    return {"baseline": baseline_text, "steered": steered_text}
