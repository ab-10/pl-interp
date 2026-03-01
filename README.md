# LLM PL Understanding

**Target demo:** interface for modifying LLM features responsible for code generation properties.

Final deliverable:

1. Running mistral model
2. SAE for identifying relevant features
3. Identified feature(s) related to code
4. Web UI for amplifying/decreasing the feature

POC: mistral 7B w/ existing SAE

VM access:
`ssh -i ~/.ssh/id_rsa azureuser@20.38.0.252`


1. open-weight Mistral 7B locally for the steered generation (where you need hooks)
2. **Mistral API** for everything that benefits from a smarter, faster model — feature labeling and prompt processing.


## Stack

| Component | New | Notes |
|---|---|---|
| Steered model | `mistralai/Mistral-7B-Instruct-v0.1` | Local, open weights (Apache 2.0) |
| SAE | `tylercosgrove/mistral-7b-sparse-autoencoder-layer16` | Community-trained, SAELens-compatible, layer 16 only |
| Feature labeling | **Mistral API** (`mistral-medium` or `mistral-small`) | Auto-interp pipeline |
| Code prompt understanding | **Mistral API** or **Codestral API** | Suggest relevant features for a given prompt |

## Custom SAE

The community SAE (`tylercosgrove/mistral-7b-sparse-autoencoder-layer16`) failed to produce meaningful steering — typing features produced zero type annotations at any strength up to +500. Root causes: 80% dead features, TopK-128 constant sparsity, MLP-output hook point, and general-text training data.

We trained a replacement BatchTopK SAE on `bigcode/starcoderdata` targeting the residual stream (`blocks.16.hook_resid_post`). Checkpoint: `~/checkpoints/code_sae_v1/` on the VM.

### SAE Quality Comparison

| Metric | Community SAE | Custom SAE |
|---|---|---|
| Dead features | 80% (104,821/131K) | **52%** (17,125/32K) |
| L0 sparsity | 128 (constant) | **102 mean** (std=117, range 23–777) |
| Decoder norms | 0.003–0.658 (237x range) | **1.0 (uniform)** |
| Hook point | `hook_mlp_out` | **`hook_resid_post`** |
| Architecture | TopK | **BatchTopK** |
| Training data | Pile (general text) | **StarCoderData (code)** |
| Training tokens | Unknown | 93M (target was 2–4B) |

The 52% dead rate is above the <10% target due to the short training run. A longer run (2–4B tokens) would improve this. However, the alive features show strong property-specific signals, and the uniform decoder norms make steering strength comparable across features.

### Discovered Features

Contrastive analysis on typed/untyped, error-handling/simple, functional/imperative, recursive/iterative, and TypeScript/JavaScript pairs:

| Property | Top Feature | Score | Positive Freq | Negative Freq | Steering Effect |
|---|---|---|---|---|---|
| TypeScript types | **304** | 0.743 | 100% | 0% | Generates typed TypeScript at +5 |
| Verbose comments | **10177** | 0.389 | 100% | 0% | Shifts output structure and elaboration |
| Recursion | **17862** | 0.289 | 100% | 33% | Structural code pattern feature |
| Functional style | **14745** | 0.201 | 67% | 0% | Lambda/higher-order function detector |
| Type annotations (Python) | **304** | 0.094 | 33% | 0% | Same feature as TS — genuine typing signal |
| Error handling | **28883** | 0.956 | 0% | 100% | "Simple code" detector (inverse signal) |

Feature 304 is the key result: it fires exclusively on typed code (TypeScript and typed Python) and, when steered at +5, causes the model to generate `function isPalindrome(input: string): boolean {` — actual type annotations. The old SAE's typing features never produced any at any strength.

---

# Milestones

## Basic Setup

**Status:** done

**Delivery criteria:**
1. Drivers installed on the GPU VM
2. Can run Steered model and SAE on the VM
3. Verify steering works with a **directional check** — amplifying a feature should produce an expected directional shift in output (e.g., boosting an "error handling" feature should yield more try/catch blocks), not just any difference

**Steps:**
1. Install NVIDIA drivers + CUDA on the GPU VM
2. Install miniconda, create Python env
3. Install PyTorch (CUDA 12), transformer_lens, sae_lens, mistralai
4. Download Mistral 7B + the Tyler Cosgrove SAE
5. Run a test: forward pass through model, encode with SAE, apply a steering hook, generate output — all in a Python script directly on the VM

## Feature Discovery

**Status:** not started

**Delivery criteria:**
1. Collect top firing features for code prompts using an **existing dataset** (The Stack, HumanEval, or similar) — no manual prompt curation
2. Label the features (using Mistral API)
3. Verify that adjusting the features, meaningfully changes code generation performance

**Note:** Tight deadline — if layer 16 is too sparse on code features, there is no time for custom SAE training. Pivot options should be evaluated early.

**Implementation discrepancies vs `FEATURE_DISCOVERY_v2.md`:**
- Scripts use **single-GPU sequential** processing (plan), not `torchrun` multi-GPU (v2). VM has 2 H100s, not 95.
- Uses **HumanEval via MultiPL-E** (8 languages, ~1,280 prompts) instead of HumanEval-only.
- Steering verification uses **side-by-side print output**, not Claude Opus LLM-as-judge. No `anthropic` dependency.
- No `00_explore_sae.py`, `00b_inspect_contrast.py`, or `config.py` — added in v2 but not in approved plan.
- Differential score formula: `code_freq * code_mean - noncode_freq * noncode_mean` (additive), not the ratio formula from v2.

## Feature Steering CLI

**Status:** not started

**Delivery criteria:**
1. A CLI version of the final web interface implemented
2. Allows specifying the features and seeing how generated code changes by toggling features on/off
    **Target behavior:** enabling features, leads to meaningful differences in the relevant code generation behavior.

## Feature Steering Web App

**Status:** not started

**Delivery criteria:**
1. A NextJS web app with a connected backend
2. User can select the features they want to boost/decrease the performance of
3. There's a textbox for entering a code generation prompt
4. Mistral model runs and generates output based on user's prompt and specified boosting features
5. The model runs on a GPU VM
    I have shell access, we can set it up together.

**Note:** Architecture (local Next.js + SSH tunnel vs. everything on VM) is **undecided** — will be determined when we reach this milestone.


# Claude Code notes

## Phase 1: Feature Discovery for Code (Updated)

Since you're limited to a single layer (16 of 32), you need to be **strategic** about finding code-relevant features within it. The process:

**Step 1 — Find firing features on code inputs:**
```python
from transformer_lens import HookedTransformer
from sae_lens import SAE

model = HookedTransformer.from_pretrained_no_processing("mistral-7b-instruct")
sae = SAE.from_pretrained(
    release="tylercosgrove/mistral-7b-sparse-autoencoder-layer16",
    sae_id="mistral-7b-sparse-autoencoder-layer16",
    device="cuda"
)

# Run code samples through and collect which features fire
code_prompts = [
    "def sort_list(items: List[int]) -> List[int]:",
    "public class Calculator { private int value;",
    "try:\n    result = x / y\nexcept ZeroDivisionError:",
    # ... 50+ varied code examples
]

feature_activations = {}  # feature_idx -> list of (prompt, activation_value)
for prompt in code_prompts:
    _, cache = model.run_with_cache(prompt)
    acts = cache["blocks.16.hook_resid_post"]
    feature_acts = sae.encode(acts)
    # Record top-firing features for this prompt
    top_features = feature_acts.topk(20).indices.tolist()
    for f in top_features:
        feature_activations.setdefault(f, []).append(prompt)
```

**Step 2 — Auto-label with Mistral API:**
```python
import mistralai

client = mistralai.Mistral(api_key=MISTRAL_API_KEY)

def label_feature(feature_idx, activating_examples):
    prompt = f"""These code snippets all strongly activate a specific internal feature of a language model.
What concept or pattern does this feature likely represent?

Examples:
{chr(10).join(activating_examples[:10])}

Respond with a short label (3-7 words) and a one-sentence description."""
    
    response = client.chat.complete(
        model="mistral-medium-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

This gives you a curated, auto-labeled feature registry — built entirely from Mistral models, which is the story you want to tell.

---

## Phase 2: Steering Backend (Updated)

Layer 16 is actually a reasonable layer for this — it's in the middle-to-late portion of Mistral 7B's 32 layers, where more semantic/behavioral features tend to live (earlier layers capture syntax, later layers capture high-level behavior).

```python
from functools import partial

def make_steering_hook(sae, feature_indices_and_strengths):
    """
    feature_indices_and_strengths: list of (feature_idx, strength) tuples
    strength > 0 = amplify, strength < 0 = suppress, strength = 0 = ablate
    """
    def hook(value, hook):
        for feat_idx, strength in feature_indices_and_strengths:
            if strength == 0:
                # Full ablation: zero out this feature's contribution
                feature_direction = sae.W_dec[feat_idx]
                projection = (value @ feature_direction).unsqueeze(-1) * feature_direction
                value = value - projection
            else:
                # Amplify/suppress by adding scaled decoder vector
                value = value + strength * sae.W_dec[feat_idx]
        return value
    return hook

def generate_with_steering(prompt, feature_overrides, max_tokens=300):
    hook_fn = make_steering_hook(sae, feature_overrides)
    with model.hooks(fwd_hooks=[("blocks.16.hook_resid_post", hook_fn)]):
        output = model.generate(prompt, max_new_tokens=max_tokens, temperature=0.3)
    return output

# Also generate baseline (no steering) for comparison
baseline = model.generate(prompt, max_new_tokens=300, temperature=0.3)
```

---

## ~~Phase 3: The "What Changed?" Layer~~ (CUT — out of scope)

~~This phase proposed using Mistral API to auto-explain diffs between baseline and steered outputs. Removed to keep scope tight.~~

---

## The Risk: Single-Layer Limitation

**Mitigation:** If during feature discovery you find the layer 16 SAE is too sparse on code features, there's a fallback: use SAELens to train a small, quick SAE yourself on code-specific data (The Stack, or TheVault) targeting the same layer. SAELens can train a usable SAE on an A100 in 2–4 hours for a 7B model.

---

## Updated Timeline

| Time | Task |
|---|---|
| Hour 1 | Load Mistral 7B + Tyler's SAE, verify steering works end-to-end in notebook |
| Hour 2–3 | Run feature discovery pipeline on 50+ code prompts, collect top firing features |
| Hour 3–4 | Auto-label features using Mistral API, curate registry of ~15 code features |
| Hour 4–5 | Build steering backend with baseline + steered side-by-side generation |
| Hour 5–6 | Build control panel UI |
| Hour 6–7 | Scripted demo prep, edge case testing |

---

## Where to Start

1. Verify the SAE loads: `pip install sae-lens transformer-lens mistralai`, then load `tylercosgrove/mistral-7b-sparse-autoencoder-layer16` and confirm `sae.encode()` works on a Mistral forward pass
2. Get a Mistral API key from [console.mistral.ai](https://console.mistral.ai) — the free tier is sufficient for feature labeling
3. Run the feature discovery pipeline on 20–30 code snippets to see what layer 16 actually captures before committing to the UI build

The core pitch is clean: **open-weight Mistral for the mechanistic steering, Mistral API for the interpretability layer** — a full Mistral stack, two different but complementary roles.

