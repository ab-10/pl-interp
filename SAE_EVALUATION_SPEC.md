# SAE Evaluation Specification

Specification for building a script that compares four custom-trained SAEs against each other and the community baseline to determine which performs best at representing and steering coding-specific features.

---

## 1. What "Best" Means

An SAE is better if it:

1. **Has more alive features** (fewer dead neurons)
2. **Finds more property-specific features** that score high on differential analysis and low on cross-property contamination
3. **Steers generation measurably** — property density increases monotonically with steering strength

These three levels form a pipeline: an SAE must pass each level to be evaluated at the next. An SAE with 90% dead features has nothing to steer with. An SAE with many alive features but no property-specific differential has no steering targets. Only SAEs that produce monotonic steering trends on specific properties are production-viable.

---

## 2. Inputs

### SAEs to evaluate

Five SAE checkpoints, loaded one at a time:

```python
SAE_CONFIGS = [
    {"name": "community",    "path_or_release": "tylercosgrove/mistral-7b-sparse-autoencoder-layer16", "is_pretrained": True},
    {"name": "custom_v1",    "path_or_release": "/path/to/checkpoints/v1/final", "is_pretrained": False},
    {"name": "custom_v2",    "path_or_release": "/path/to/checkpoints/v2/final", "is_pretrained": False},
    {"name": "custom_v3",    "path_or_release": "/path/to/checkpoints/v3/final", "is_pretrained": False},
    {"name": "custom_v4",    "path_or_release": "/path/to/checkpoints/v4/final", "is_pretrained": False},
]
```

Loading logic must handle both cases:

```python
from sae_lens import SAE

if config["is_pretrained"]:
    # Community SAE from HuggingFace
    sae = SAE.from_pretrained(release=config["path_or_release"], sae_id=".")[0]
else:
    # Custom-trained SAE from local checkpoint
    sae = SAE.load_from_disk(config["path_or_release"], device=device)
```

### Shared model

All SAEs are evaluated against the same base model:

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained_no_processing(
    "mistralai/Mistral-7B-Instruct-v0.1",
    dtype=torch.float16,
)
```

The hook point comes from each SAE's metadata:

```python
hook_point = sae.cfg.metadata.get("hook_name", "blocks.16.hook_resid_post")
```

Different SAEs may use different hook points (community uses `blocks.16.hook_mlp_out`; custom ones use `blocks.16.hook_resid_post`). This is expected and correct.

---

## 3. Level 1: SAE Health Profile

### What to measure

For each SAE, compute these metrics by running a fixed set of diverse prompts through the model and encoding the activations:

| Metric | How to compute | What it tells you |
|---|---|---|
| `d_sae` | `sae.cfg.d_sae` | Dictionary size |
| `dead_fraction` | Fraction of features that never fire (`activation > 0`) on any token across all prompts | How much of the dictionary is wasted |
| `alive_count` | `d_sae - dead_count` | Usable capacity |
| `l0_mean` | Mean number of features with `activation > 0` per token | Sparsity level |
| `l0_std` | Standard deviation of L0 across tokens | L0 variance (0 means constant like TopK; >0 means natural variation like BatchTopK) |
| `decoder_norm_mean` | `sae.W_dec.norm(dim=1).mean()` | Average magnitude of steering vectors |
| `decoder_norm_std` | `sae.W_dec.norm(dim=1).std()` | How uniform the steering vectors are |

### Prompt set for health profiling

Use the same prompt set for all SAEs. Include both code (various languages and styles) and prose (for contrast). At minimum, 30 diverse prompts covering:
- Python (typed, untyped, error handling, functional, imperative, recursive)
- TypeScript, JavaScript, Rust, C++
- Prose (history, science, instructions)

### Algorithm

```
for each SAE:
    ever_fired = zeros(d_sae, dtype=bool)
    l0_values = []

    for each prompt in prompt_set:
        tokens = model.to_tokens(prompt, prepend_bos=True)
        tokens = tokens[:, :512]  # truncate to avoid OOM
        _, cache = model.run_with_cache(tokens, names_filter=hook_point)
        acts = cache[hook_point]  # [1, seq_len, d_model]

        for pos in range(seq_len):
            feat_acts = sae.encode(acts[0, pos, :].float().unsqueeze(0)).squeeze(0)
            active = feat_acts > 0
            ever_fired |= active.cpu()
            l0_values.append(active.sum().item())

    dead_count = (~ever_fired).sum().item()
    dead_fraction = dead_count / d_sae
    l0_mean = mean(l0_values)
    l0_std = std(l0_values)
```

### Scoring

```
health_score = alive_fraction * (1 + min(l0_std, 10) / 10)
```

Rationale: An SAE with more alive features is better. An SAE with natural L0 variance (BatchTopK) gets a bonus over constant-L0 (TopK), because variable sparsity means the dictionary adapts to input complexity. The `min(l0_std, 10)/10` term gives a 0-to-1 bonus for L0 variance, capped at std=10.

---

## 4. Level 2: Feature Discovery via Differential Analysis

### Contrastive code pairs

Use pre-written code snippets (not model-generated) to remove generation variance. Each property has a positive set and a negative set.

Six properties to test:

**Property 1: Type annotations**
- Positive: Python functions with full type hints (`def foo(x: int) -> str:`)
- Negative: Same functions without type hints (`def foo(x):`)

**Property 2: Error handling**
- Positive: Code with `try/except/catch/finally` blocks
- Negative: Same logic without error handling

**Property 3: Functional style**
- Positive: Code using `map`, `filter`, `reduce`, `lambda`, list comprehensions
- Negative: Same logic using `for` loops and mutation

**Property 4: Recursion**
- Positive: Recursive implementations (fibonacci, tree traversal, flatten)
- Negative: Iterative implementations of the same algorithms

**Property 5: TypeScript types**
- Positive: TypeScript code with interfaces and type annotations
- Negative: Equivalent JavaScript without types

**Property 6: Verbose documentation**
- Positive: Code with extensive comments and docstrings on every line
- Negative: Same code with no comments

For each property, provide **at least 3 positive and 3 negative examples** (more is better). The positive and negative sets must be matched — same algorithm/functionality, differing only in the target property.

### Algorithm

For each SAE, for each property:

```
positive_acts = []  # list of [d_sae] tensors
negative_acts = []

for each positive snippet:
    tokens = model.to_tokens(snippet, prepend_bos=False)[:, :512]
    _, cache = model.run_with_cache(tokens, names_filter=hook_point)
    acts = cache[hook_point]
    mean_acts = acts[0].mean(dim=0).float().unsqueeze(0)
    feat_acts = sae.encode(mean_acts).squeeze(0)  # [d_sae]
    positive_acts.append(feat_acts)

# same for negative snippets
positive_acts = stack(positive_acts)   # [n_pos, d_sae]
negative_acts = stack(negative_acts)   # [n_neg, d_sae]

pos_mean = positive_acts.mean(dim=0)   # [d_sae]
neg_mean = negative_acts.mean(dim=0)   # [d_sae]
pos_freq = (positive_acts > 0).float().mean(dim=0)  # [d_sae]
neg_freq = (negative_acts > 0).float().mean(dim=0)  # [d_sae]
```

### Scoring formula

For each feature, compute a differential score:

```
diff = pos_mean - neg_mean
score = diff * (pos_freq - neg_freq + 0.1)
```

The `+0.1` prevents zeroing out features where frequency is equal but mean activation differs. This is the formula from `explore_trained_sae.py`.

Take the top 10 features per property by `score` (descending). These are the property's **candidate features**.

### Aggregate metrics per SAE

For each property, record:

| Metric | Definition |
|---|---|
| `top1_score` | Score of the best feature for this property |
| `top5_mean_score` | Mean score of the top 5 features |
| `top1_specificity` | `pos_mean[best] / (neg_mean[best] + 1e-8)` — how exclusive is the best feature? |
| `top1_pos_freq` | How consistently the best feature fires on positive examples |

### Cross-property contamination check

After computing top-5 features for all 6 properties, check how many features appear in the top-5 of **more than one** property. A good SAE has zero or minimal cross-contamination — each feature should be specific to one property.

```
contamination_count = number of features appearing in top-5 of multiple properties
contamination_fraction = contamination_count / total unique features in all top-5 lists
```

### Discovery score per SAE

```
per_property_score = mean(top1_score across all 6 properties)
specificity_bonus = mean(top1_specificity across all 6 properties)
contamination_penalty = 1.0 - contamination_fraction

discovery_score = per_property_score * specificity_bonus * contamination_penalty
```

---

## 5. Level 3: Steering Verification

### Which features to steer

For each SAE, steer the **top-1 feature per property** from Level 2. That's 6 features per SAE.

### Steering mechanism

The steering hook adds a scaled copy of the SAE's decoder vector for the target feature to the residual stream at the hook point during generation:

```python
steering_vector = sae.W_dec[feature_idx].detach().clone()

def steering_hook(value, hook):
    value[:, :, :] = value + strength * steering_vector.to(value.device, value.dtype)
    return value

tokens = model.to_tokens(prompt, prepend_bos=False)
with model.hooks(fwd_hooks=[(hook_point, steering_hook)]):
    output_tokens = model.generate(
        tokens,
        max_new_tokens=200,
        temperature=0.0,  # greedy for reproducibility
        do_sample=False,
    )
output = model.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
```

### Strength sweep

Test 7 strengths: `[-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]`

At `strength=0.0`, generate a baseline (no hook).

### Prompts

Use ambiguous prompts that don't explicitly request any property. This lets the steering effect show:

```python
EVAL_PROMPTS = [
    "Write a Python function that takes a list and returns the largest element",
    "Implement a stack data structure with push, pop, and peek methods",
    "Write a function that merges two sorted arrays into one sorted array",
    "Create a function that validates an email address",
]
```

### Property density measurement

For each generated output, count regex pattern matches to measure how much of the target property is present. Density = total_matches / total_lines.

**Type annotations:**
```python
[
    r":\s*(?:int|float|str|bool|list|dict|tuple|set|None|Any|Optional|Union|Callable)\b",
    r"->\s*(?:int|float|str|bool|list|dict|tuple|set|None|Any|Optional|Union)\b",
    r":\s*(?:List|Dict|Tuple|Set)\[",
    r":\s*(?:number|string|boolean|void|never|any|unknown)\b",
    r":\s*(?:Array|Map|Set|Record|Promise|Partial)\s*<",
    r"\binterface\s+\w+",
    r"\btype\s+\w+\s*=",
]
```

**Error handling:**
```python
[
    r"\btry\b",
    r"\bexcept\b",
    r"\bcatch\b",
    r"\bfinally\b",
    r"\braise\b",
    r"\bthrow\b",
]
```

**Functional style:**
```python
[
    r"\bmap\s*\(",
    r"\bfilter\s*\(",
    r"\breduce\s*\(",
    r"\blambda\b",
    r"\.\bmap\b",
    r"\.\bfilter\b",
    r"\.\breduce\b",
    r"\[.+\bfor\b.+\bin\b.+\]",
]
```

**Recursion:**
```python
[
    r"\breturn\s+\w+\s*\(",
    r"\brecurs",
]
```

**TypeScript types** — same as "Type annotations" patterns.

**Verbose documentation:**
```python
[
    r'"""',
    r"'''",
    r"#\s+\S",
    r"//\s+\S",
    r"/\*",
    r"\bArgs:\b",
    r"\bReturns:\b",
    r"\bExample",
]
```

For each property, compute density at each strength. Formula:

```python
import re

def compute_density(text, patterns):
    lines = text.strip().split("\n")
    total_lines = max(len(lines), 1)
    total_matches = sum(len(re.findall(p, text, re.MULTILINE | re.IGNORECASE)) for p in patterns)
    return total_matches / total_lines
```

### Monotonicity test

For each feature, average density across all 4 prompts at each strength level. Then check:

```python
neg_avg = mean(density at strengths < 0)
baseline = density at strength 0.0
pos_avg = mean(density at strengths > 0)

is_monotonic = (pos_avg > baseline) and (baseline > neg_avg)
```

A feature **passes** if `is_monotonic` is `True`.

### Coherence check

At the strongest positive strength (+5.0), check that the output hasn't degenerated into gibberish. Simple heuristic: the output must contain at least one recognizable code keyword (`def`, `function`, `class`, `return`, `if`, `for`, `while`, `import`, `const`, `let`, `var`).

```python
CODE_KEYWORDS = r"\b(def|function|class|return|if|for|while|import|const|let|var)\b"
is_coherent = bool(re.search(CODE_KEYWORDS, output))
```

### Steering score per SAE

```
For each property:
    monotonic = 1 if is_monotonic else 0
    coherent = 1 if is_coherent at +5.0 else 0
    effect_size = pos_avg - neg_avg  (larger = stronger steering effect)

    property_steering_score = monotonic * coherent * effect_size

steering_score = mean(property_steering_score across all 6 properties)
```

---

## 6. Final Ranking

Combine the three levels into a single ranking. Each level acts as a gate: an SAE that fails early levels can't recover on later ones.

```
final_score = health_score * discovery_score * (1 + steering_score)
```

The `(1 + steering_score)` term means:
- An SAE with zero monotonic features gets a 1x multiplier (no bonus, no penalty from steering)
- An SAE with strong steering gets >1x multiplier (rewarded)

This prevents an SAE with good health and discovery from being zeroed out just because steering is weak (which may reflect SAE quality OR insufficient steering strength OR property-detection-vs-generation mismatch).

### Output format

```json
{
  "ranking": [
    {
      "name": "custom_v2",
      "final_score": 1.847,
      "health_score": 0.93,
      "discovery_score": 2.15,
      "steering_score": 0.42,
      "health": {
        "d_sae": 32768,
        "dead_fraction": 0.07,
        "alive_count": 30474,
        "l0_mean": 63.8,
        "l0_std": 12.4
      },
      "discovery": {
        "per_property": {
          "type_annotations": {"top1_feature": 4521, "top1_score": 3.42, "top1_specificity": 18.7, "top1_pos_freq": 1.0},
          "error_handling": {"top1_feature": 8821, "top1_score": 2.87, "top1_specificity": 12.3, "top1_pos_freq": 0.67},
          ...
        },
        "contamination_fraction": 0.0,
        "discovery_score": 2.15
      },
      "steering": {
        "per_property": {
          "type_annotations": {"feature": 4521, "monotonic": true, "coherent": true, "neg_avg": 0.0, "baseline": 0.02, "pos_avg": 0.35, "effect_size": 0.35},
          ...
        },
        "steering_score": 0.42
      }
    },
    ...
  ]
}
```

Sort the `ranking` array by `final_score` descending. The first entry is the winner.

---

## 7. Implementation Notes

### Memory management

Only one SAE should be loaded at a time. After finishing all three levels for one SAE, delete it and free VRAM before loading the next:

```python
del sae
torch.cuda.empty_cache()
```

The model stays loaded throughout (it's shared).

### Determinism

Use `temperature=0.0` and `do_sample=False` for steering verification so outputs are reproducible across SAEs. For the health profile and feature discovery steps, determinism isn't needed (those don't involve generation).

### Hook point varies per SAE

The community SAE uses `blocks.16.hook_mlp_out`. Custom SAEs may use `blocks.16.hook_resid_post` or other hook points. Always read the hook point from the SAE's metadata rather than hardcoding it:

```python
hook_point = sae.cfg.metadata.get("hook_name", "blocks.16.hook_resid_post")
```

### SAE encoding dtype

SAE encoding requires float32 input regardless of the model's dtype:

```python
acts = cache[hook_point]  # float16
feat_acts = sae.encode(acts.float())  # must cast to float32
```

### Expected runtime

Per SAE (rough estimates on a single H100):
- Level 1 (health): ~5 minutes (30 prompts, per-token encoding)
- Level 2 (discovery): ~3 minutes (36 snippets, mean-pooled encoding)
- Level 3 (steering): ~30 minutes (6 features x 4 prompts x 7 strengths x 200 tokens generation)

Total for 5 SAEs: ~3-4 hours. Level 3 dominates. To cut time, reduce `max_new_tokens` to 150 or test fewer prompts.
