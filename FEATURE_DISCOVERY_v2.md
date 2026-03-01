# Plan: Milestone 2 — Feature Discovery

## Context

Milestone 1 is complete. Scripts live in `scripts/` and run on the GPU VM under `conda activate steering`. The VM has **95 × H100 (80 GB)** instances available.

---

## Delivery Criteria

1. Collect top-firing features for code prompts using **MultiPL-E** (multilingual)
2. Label features using the Mistral API
3. Verify steering using **Claude Opus** as an automated judge

---

## Scripts

### `scripts/requirements.txt`
```
datasets
mistralai
anthropic
accelerate
```

---

### `scripts/00_explore_sae.py` *(new)*

**Purpose:** Before running any feature discovery, characterize the SAE so downstream decisions (K, batch size, steering strength) are grounded in data rather than guesses.

**What to compute and print:**

| Property | How to Compute | Why It Matters |
|---|---|---|
| Dictionary size (M) | `sae.W_dec.shape[0]` | Sets the ceiling on K; K=50 is meaningless if M=512 |
| Dead features | Run 500 random prompts; count features that never fire | High dead-feature rate signals the SAE may be undertrained on code |
| Mean L0 sparsity | Average number of features that fire per token | If L0 is very low (e.g. <5), K=50 captures mostly noise |
| Feature activation distribution | Histogram of max activation per feature across the 500 prompts | Identifies whether a few features dominate or activations are spread |
| Decoder vector norms | `sae.W_dec.norm(dim=1)` stats (min/max/mean) | Unnormalized decoders will make steering strength non-comparable across features |

**Output:** `results/sae_profile.json` + printed summary table. This script must be reviewed before proceeding to script 01, and K and steering strength should be updated in a shared `config.py` accordingly.

---

### `scripts/config.py` *(new, shared constants)*
```python
# Set after reviewing 00_explore_sae.py output
TOP_K = 50                  # features recorded per prompt — revisit if L0 < 10
N_GPUS = 95
BATCH_SIZE_PER_GPU = 48     # see calculation below
STEERING_STRENGTHS = [1.0, 3.0, 10.0]   # sweep, not a single value
TOP_FEATURES_TO_LABEL = 20
JUDGE_MODEL = "claude-opus-4-6"
MISTRAL_LABEL_MODEL = "mistral-medium-latest"
```

**Batch size rationale:**
- H100 VRAM: 80 GB
- Mistral 7B in bf16: ~14 GB
- Remaining for activations + KV cache: ~66 GB
- Layer 16 residual stream activations per token: `4096 × 2 bytes = 8 KB`
- At sequence length 256 tokens, one sample's activations ≈ 2 MB
- Conservative headroom for KV cache and gradients: 30 GB reserved
- Usable for batch: ~36 GB → **~48 sequences per GPU** at seq len 256
- Across 95 GPUs: **~4,560 sequences processed per forward pass wave**

This is a starting estimate. Run `scripts/00_explore_sae.py` first and adjust if OOM errors occur.

---

### `scripts/01_collect_activations.py`

**Purpose:** Run code and non-code prompts through Mistral 7B in parallel across all 95 H100s, extract layer 16 SAE activations, record top-K firing features per prompt.

#### Dataset: MultiPL-E (replaces HumanEval)

MultiPL-E supports 22 programming languages, translating HumanEval and MBPP benchmarks using language-specific transpilers. Load the following language subsets from `nuprl/MultiPL-E` on HuggingFace:

```python
LANGUAGES = ["py", "js", "java", "cpp", "rs", "ts", "sh", "r"]
# Covers: scripting, systems, typed OOP, functional-adjacent, shell
# ~160 problems × 8 languages = ~1,280 code prompts
```

This directly addresses the Python-only limitation of HumanEval and means discovered features must generalize across language paradigms to rank highly — a much stronger signal.

#### Contrast Set (non-code): See dedicated section below

#### Parallelization

Use `torch.distributed` with one process per GPU:

```python
# Launch with:
# torchrun --nproc_per_node=95 01_collect_activations.py

import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()          # which GPU (0–94)
world_size = dist.get_world_size()  # 95

# Shard dataset across GPUs — each GPU processes its slice independently
prompts_for_this_gpu = all_prompts[rank::world_size]

# Load model on this GPU
model = HookedTransformer.from_pretrained_no_processing(
    "mistralai/Mistral-7B-Instruct-v0.1",
    dtype=torch.bfloat16,
    device=f"cuda:{rank}"
)

# Process in batches of BATCH_SIZE_PER_GPU
for batch in chunked(prompts_for_this_gpu, BATCH_SIZE_PER_GPU):
    tokens = model.to_tokens(batch, prepend_bos=True)
    _, cache = model.run_with_cache(tokens, names_filter="blocks.16.hook_resid_post")
    acts = cache["blocks.16.hook_resid_post"]   # [batch, seq, d_model]
    # Mean-pool over sequence dimension before SAE encode
    acts_pooled = acts.mean(dim=1)              # [batch, d_model]
    feature_acts = sae.encode(acts_pooled)      # [batch, M]
    # Record top-K per prompt ...

# Each rank writes its shard: results/activations_code_rank{rank}.json
# A final gather step merges shards on rank 0
```

**Output:** `results/activations_code.json`, `results/activations_noncode.json`

---

### `scripts/02_find_code_features.py`

**Purpose:** Differential analysis — find features that fire strongly and consistently on code across *multiple languages*, but not on plain prose.

- For each feature compute:
  - `freq_code`: fraction of code prompts where it fired
  - `mean_act_code`: mean activation value on code prompts
  - `freq_noncode`: fraction of non-code prompts where it fired
  - `differential_score = (mean_act_code × freq_code) / (mean_act_noncode × freq_noncode + ε)`
- Secondary filter: **cross-language consistency** — prefer features that fire across ≥3 of the 8 loaded languages, not just Python. This surfaces semantically general code features rather than Python-specific syntax tokens.

**Output:** `results/code_features_ranked.json` — top 50 features with full stats, language breakdown, and cross-language consistency score.

---

### `scripts/03_label_features.py`

No changes to the core logic. One addition: include the **language distribution** of activating examples in the Mistral prompt, so the label reflects cross-language generality where applicable.

```python
prompt = f"""These code snippets (across multiple programming languages) all strongly activate
a specific internal feature of a language model. What concept or pattern does this feature represent?

Examples:
{examples_with_language_tags}

Language distribution: {lang_dist}

Respond with a short label (3-7 words) and a one-sentence description."""
```

**Output:** `results/feature_registry.json`

---

### `scripts/04_verify_steering.py`

**Purpose:** Automated directional verification using **Claude Opus** (`claude-opus-4-6`) as the judge, replacing manual review.

#### Steering sweep

For each labeled feature, generate outputs at **multiple steering strengths** rather than a single value:

```python
for feat in top_features:
    baseline = generate(prompt, steering=[])
    results = {
        "baseline": baseline,
        "steered": {}
    }
    for strength in STEERING_STRENGTHS:  # [1.0, 3.0, 10.0]
        steered = generate(prompt, steering=[(feat["index"], strength)])
        results["steered"][strength] = steered
```

#### LLM-as-judge prompt

```python
import anthropic
client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY from env

judge_prompt = f"""You are evaluating whether a language model's output has shifted in a specific direction.

Feature label: "{feature['label']}"
Feature description: "{feature['description']}"
Steering strength: {strength}

Baseline output:
{baseline}

Steered output:
{steered}

Questions:
1. Does the steered output exhibit more of the behavior described by the feature label than the baseline? (yes/no/unclear)
2. Does the steered output remain coherent and syntactically valid code? (yes/no/degraded)
3. Rate the directional shift on a scale of 0–5, where 0 = no change, 5 = clear strong shift.
4. One sentence describing what concretely changed.

Respond in JSON."""

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=512,
    messages=[{"role": "user", "content": judge_prompt}]
)
```

The judge verdict per (feature, strength) tuple is saved to `results/steering_verification.json`. The summary printed to stdout should include: for each feature, the minimum strength at which a directional shift is confirmed, and whether coherence degrades at higher strengths.

This replaces the manual review step with a consistent, repeatable, logged evaluation — and the steering strength sweep means you discover the effective operating range for each feature rather than committing blindly to +3.0.

---

## Contrast Set Specification *(detailed)*

This was underspecified in the original plan. The contrast set's quality directly determines how clean the differential signal is. Requirements:

**What the contrast set must be:**

- **Pure declarative prose** — factual sentences with no structure. Examples: encyclopedia-style paragraphs, general knowledge Q&A, narrative text.
- **No lists, no numbered steps, no pseudocode, no technical jargon** — these can share structural features with code even without being code.
- **No math or logical notation** — symbolic reasoning patterns may overlap with features that fire on typed languages.
- **Linguistically diverse** — vary sentence length, topic domain (history, geography, biology, culture), and register (formal vs. informal prose).
- **~300 prompts** (larger than the code set) to give the differential analysis statistical power.

**Recommended source:** `tatsu-lab/alpaca` filtered to instruction-response pairs where the instruction contains none of: `code`, `function`, `script`, `program`, `algorithm`, `write a`, `implement`. Apply a secondary regex filter to exclude any response that contains backticks, indentation blocks, or `=` assignments.

**Required data exploration before running script 01:**

Add a quick `00b_inspect_contrast.py` that prints:
- Average token length of contrast prompts vs. code prompts (they should be comparable — very short contrast prompts will bias frequency stats)
- Whether any contrast prompts contain code-adjacent tokens (backtick, `def`, `{`, `;`) — flag and remove
- Vocabulary overlap between contrast and code sets — if overlap is too high (>60% of unique tokens), the contrast set is too similar to code to be useful

**What to avoid:** Do not use Alpaca without filtering, do not use any dataset that includes instruction-following on structured tasks (e.g., "write a recipe," "list the steps"), and do not use a contrast set that is substantially shorter in token length than the code prompts, as this will artificially suppress non-code activation values.

---

## File Summary

| File | Purpose |
|---|---|
| `scripts/config.py` | Shared constants (K, batch size, steering strengths, model names) |
| `scripts/requirements.txt` | Additional VM Python deps |
| `scripts/00_explore_sae.py` | Characterize SAE before committing to parameters |
| `scripts/00b_inspect_contrast.py` | Validate contrast set quality before running pipeline |
| `scripts/01_collect_activations.py` | Parallel activation collection across 95 H100s |
| `scripts/02_find_code_features.py` | Differential + cross-language feature ranking |
| `scripts/03_label_features.py` | Mistral API auto-labeling with language context |
| `scripts/04_verify_steering.py` | Strength sweep + Claude Opus judge |

---

## Verification Sequence

1. Run `00_explore_sae.py` → review SAE profile, update `config.py`
2. Run `00b_inspect_contrast.py` → validate and clean contrast set
3. `torchrun --nproc_per_node=95 01_collect_activations.py`
4. Run `02_find_code_features.py` → inspect differential scores and cross-language consistency
5. Run `03_label_features.py` → review that labels are interpretable
6. Run `04_verify_steering.py` → inspect judge verdicts; a feature is "verified" if Claude Opus rates directional shift ≥3/5 at any tested strength while coherence remains intact
7. If fewer than 5 features pass verification → trigger pivot discussion per README
