# Implement: SAE Comparative Evaluation Script

Write a single Python script `evaluate_saes.py` that compares five sparse autoencoders (four custom-trained, one community baseline) to determine which best represents and steers coding-specific features in Mistral 7B. The script runs three evaluation levels in sequence, produces a ranked JSON output, and prints a summary table.

---

## Environment

The script runs on a machine with NVIDIA H100 GPUs, CUDA, and these packages pre-installed:

- `torch` (2.6+, CUDA-enabled)
- `transformer_lens` (2.17+)
- `sae_lens` (6.37+)
- `numpy`

---

## SAE Loading

The script accepts SAE checkpoint paths via CLI arguments. There are two loading paths depending on the SAE source:

```python
from sae_lens import SAE

# Community SAE — hosted on HuggingFace
sae = SAE.from_pretrained(
    release="tylercosgrove/mistral-7b-sparse-autoencoder-layer16",
    sae_id=".",
)[0]

# Custom-trained SAE — local checkpoint directory
sae = SAE.load_from_disk("/path/to/checkpoint/final", device=device)
```

Each SAE stores its hook point in metadata. Always read it rather than hardcoding:

```python
hook_point = sae.cfg.metadata.get("hook_name", "blocks.16.hook_resid_post")
```

The community SAE uses `blocks.16.hook_mlp_out`; the custom SAEs use `blocks.16.hook_resid_post`. Both are valid — the evaluation must use whatever hook point each SAE specifies.

---

## Model Loading

All SAEs are evaluated against one shared base model. Load it once and reuse across all SAE evaluations:

```python
from transformer_lens import HookedTransformer
import torch

model = HookedTransformer.from_pretrained_no_processing(
    "mistralai/Mistral-7B-Instruct-v0.1",
    dtype=torch.float16,
)
```

---

## Activation Extraction

This is the core operation used by Levels 1 and 2. Given a text string, run it through the model, extract the residual stream at the hook point, and encode it through the SAE:

```python
tokens = model.to_tokens(text, prepend_bos=True)
if tokens.shape[1] > 512:
    tokens = tokens[:, :512]

with torch.no_grad():
    _, cache = model.run_with_cache(tokens, names_filter=hook_point)

acts = cache[hook_point]  # shape: [1, seq_len, d_model]

# SAE encoding REQUIRES float32 input, regardless of model dtype
feat_acts = sae.encode(acts[0, pos, :].float().unsqueeze(0)).squeeze(0)  # [d_sae]
```

A feature is "firing" when its activation is `> 0`.

---

## Memory Management

Only one SAE should be in GPU memory at a time. After evaluating one SAE across all three levels, delete it and clear VRAM before loading the next:

```python
del sae
torch.cuda.empty_cache()
```

---

## Level 1: SAE Health Profile

For each SAE, measure basic quality metrics by running a fixed prompt set through the model and encoding every token position through the SAE.

### Prompt set

Use all prompts from the `HEALTH_PROMPTS` list below. These cover code in multiple languages and styles, plus prose for contrast:

```python
HEALTH_PROMPTS = [
    # Python — typed
    "def binary_search(arr: list[int], target: int) -> int:\n    low: int = 0\n    high: int = len(arr) - 1\n    while low <= high:\n        mid: int = (low + high) // 2\n        if arr[mid] == target:\n            return mid",
    "from typing import Optional, Dict\n\ndef get_user(user_id: int) -> Optional[Dict[str, str]]:\n    users: Dict[int, Dict[str, str]] = {}\n    return users.get(user_id)",
    "class Stack:\n    def __init__(self) -> None:\n        self._items: list[int] = []\n    def push(self, item: int) -> None:\n        self._items.append(item)\n    def pop(self) -> int:\n        return self._items.pop()",
    # Python — untyped
    "def binary_search(arr, target):\n    low = 0\n    high = len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid",
    "def get_user(user_id):\n    users = {}\n    return users.get(user_id)",
    "class Stack:\n    def __init__(self):\n        self._items = []\n    def push(self, item):\n        self._items.append(item)\n    def pop(self):\n        return self._items.pop()",
    # Python — error handling
    "try:\n    result = int(user_input)\nexcept ValueError as e:\n    print(f'Invalid input: {e}')\n    result = 0\nexcept TypeError:\n    raise",
    "def safe_divide(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        logger.error('Division by zero')\n        return None\n    finally:\n        cleanup()",
    "try:\n    with open(filepath) as f:\n        data = json.load(f)\nexcept FileNotFoundError:\n    data = {}\nexcept json.JSONDecodeError as e:\n    raise ValueError(f'Corrupt file: {e}')",
    # Python — no error handling
    "result = int(user_input)\nprint(result)",
    "def divide(a, b):\n    return a / b",
    "with open(filepath) as f:\n    data = json.load(f)",
    # Python — functional
    "result = list(map(lambda x: x * 2, filter(lambda x: x > 0, numbers)))\ntotal = reduce(lambda a, b: a + b, result)",
    "pipeline = compose(normalize, tokenize, filter_stopwords, stem)\nprocessed = [pipeline(doc) for doc in documents]",
    "from functools import reduce\nword_counts = reduce(lambda acc, w: {**acc, w: acc.get(w, 0) + 1}, words, {})",
    # Python — imperative
    "result = []\nfor x in numbers:\n    if x > 0:\n        result.append(x * 2)\ntotal = 0\nfor r in result:\n    total += r",
    "processed = []\nfor doc in documents:\n    doc = normalize(doc)\n    doc = tokenize(doc)\n    doc = filter_stopwords(doc)\n    processed.append(doc)",
    "word_counts = {}\nfor w in words:\n    if w in word_counts:\n        word_counts[w] += 1\n    else:\n        word_counts[w] = 1",
    # Python — recursive
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
    "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result",
    "def tree_depth(node):\n    if node is None:\n        return 0\n    return 1 + max(tree_depth(node.left), tree_depth(node.right))",
    # Python — iterative
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "def flatten(lst):\n    stack = [lst]\n    result = []\n    while stack:\n        current = stack.pop()\n        if isinstance(current, list):\n            stack.extend(reversed(current))\n        else:\n            result.append(current)\n    return result",
    # TypeScript
    "interface User {\n  id: number;\n  name: string;\n  email: string;\n}\n\nfunction getUser(id: number): User | undefined {\n  const users: User[] = [];\n  return users.find(u => u.id === id);\n}",
    "type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };\n\nfunction parseJson<T>(input: string): Result<T, Error> {\n  try {\n    return { ok: true, value: JSON.parse(input) };\n  } catch (e) {\n    return { ok: false, error: e as Error };\n  }\n}",
    # JavaScript
    "function getUser(id) {\n  const users = [];\n  return users.find(u => u.id === id);\n}",
    "function parseJson(input) {\n  try {\n    return { ok: true, value: JSON.parse(input) };\n  } catch (e) {\n    return { ok: false, error: e };\n  }\n}",
    # Verbose comments
    "# This function performs a binary search on a sorted array.\n# It takes an array and a target value as input.\n# Returns the index of the target if found, otherwise -1.\ndef binary_search(arr, target):\n    # Initialize the low and high pointers\n    low = 0  # Start of the array\n    high = len(arr) - 1  # End of the array\n    # Continue searching while the search space is valid\n    while low <= high:\n        mid = (low + high) // 2  # Calculate the middle index\n        if arr[mid] == target:  # Found the target\n            return mid",
    # Minimal comments
    "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
    # Prose
    "The French Revolution began in 1789 with the storming of the Bastille.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The theory of general relativity describes gravity as spacetime curvature.",
]
```

### Metrics to compute

For each SAE, iterate over every prompt. For each prompt, tokenize it (`prepend_bos=True`), truncate to 512 tokens, run through the model, and encode **each token position** through the SAE individually.

Track:

1. **`ever_fired`**: A boolean vector of length `d_sae`. Set bit `i` to `True` if feature `i` has activation `> 0` at any token position across all prompts.
2. **`l0_values`**: A list of integers. For each token position, count how many features have activation `> 0`. Append that count.
3. **Decoder norms**: `sae.W_dec.norm(dim=1)` — a vector of length `d_sae`. Compute `.mean()` and `.std()`.

After all prompts:

```python
dead_count = (~ever_fired).sum().item()
alive_fraction = 1.0 - (dead_count / d_sae)
l0_mean = numpy.mean(l0_values)
l0_std = numpy.std(l0_values)
```

### Health score formula

```
health_score = alive_fraction * (1 + min(l0_std, 10) / 10)
```

`alive_fraction` is the primary signal: more alive features means more capacity. The second term awards a 0–1 bonus for L0 variance — BatchTopK SAEs naturally vary how many features fire per token (high `l0_std`), while TopK SAEs have constant L0 (`l0_std ≈ 0`). Variable sparsity is better because the dictionary adapts to input complexity. The bonus is capped at `l0_std = 10`.

---

## Level 2: Feature Discovery via Differential Analysis

For each SAE, find features that are specific to one coding property and not others. Use pre-written contrastive code pairs (not model-generated) so activation differences are caused by the property, not by generation variance.

### Contrastive pairs

Six properties. Each has a positive set (code exhibiting the property) and a negative set (same algorithms without the property). These are the exact strings to use:

```python
CONTRASTIVE_PAIRS = [
    {
        "name": "type_annotations",
        "positive": [
            "def binary_search(arr: list[int], target: int) -> int:\n    low: int = 0\n    high: int = len(arr) - 1\n    while low <= high:\n        mid: int = (low + high) // 2\n        if arr[mid] == target:\n            return mid",
            "from typing import Optional, Dict\n\ndef get_user(user_id: int) -> Optional[Dict[str, str]]:\n    users: Dict[int, Dict[str, str]] = {}\n    return users.get(user_id)",
            "class Stack:\n    def __init__(self) -> None:\n        self._items: list[int] = []\n    def push(self, item: int) -> None:\n        self._items.append(item)\n    def pop(self) -> int:\n        return self._items.pop()",
        ],
        "negative": [
            "def binary_search(arr, target):\n    low = 0\n    high = len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid",
            "def get_user(user_id):\n    users = {}\n    return users.get(user_id)",
            "class Stack:\n    def __init__(self):\n        self._items = []\n    def push(self, item):\n        self._items.append(item)\n    def pop(self):\n        return self._items.pop()",
        ],
    },
    {
        "name": "error_handling",
        "positive": [
            "try:\n    result = int(user_input)\nexcept ValueError as e:\n    print(f'Invalid input: {e}')\n    result = 0\nexcept TypeError:\n    raise",
            "def safe_divide(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        logger.error('Division by zero')\n        return None\n    finally:\n        cleanup()",
            "try:\n    with open(filepath) as f:\n        data = json.load(f)\nexcept FileNotFoundError:\n    data = {}\nexcept json.JSONDecodeError as e:\n    raise ValueError(f'Corrupt file: {e}')",
        ],
        "negative": [
            "result = int(user_input)\nprint(result)",
            "def divide(a, b):\n    return a / b",
            "with open(filepath) as f:\n    data = json.load(f)",
        ],
    },
    {
        "name": "functional_style",
        "positive": [
            "result = list(map(lambda x: x * 2, filter(lambda x: x > 0, numbers)))\ntotal = reduce(lambda a, b: a + b, result)",
            "pipeline = compose(normalize, tokenize, filter_stopwords, stem)\nprocessed = [pipeline(doc) for doc in documents]",
            "from functools import reduce\nword_counts = reduce(lambda acc, w: {**acc, w: acc.get(w, 0) + 1}, words, {})",
        ],
        "negative": [
            "result = []\nfor x in numbers:\n    if x > 0:\n        result.append(x * 2)\ntotal = 0\nfor r in result:\n    total += r",
            "processed = []\nfor doc in documents:\n    doc = normalize(doc)\n    doc = tokenize(doc)\n    doc = filter_stopwords(doc)\n    processed.append(doc)",
            "word_counts = {}\nfor w in words:\n    if w in word_counts:\n        word_counts[w] += 1\n    else:\n        word_counts[w] = 1",
        ],
    },
    {
        "name": "recursion",
        "positive": [
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
            "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result",
            "def tree_depth(node):\n    if node is None:\n        return 0\n    return 1 + max(tree_depth(node.left), tree_depth(node.right))",
        ],
        "negative": [
            "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
            "def flatten(lst):\n    stack = [lst]\n    result = []\n    while stack:\n        current = stack.pop()\n        if isinstance(current, list):\n            stack.extend(reversed(current))\n        else:\n            result.append(current)\n    return result",
            "def tree_depth(root):\n    if root is None:\n        return 0\n    queue = [(root, 1)]\n    max_depth = 0\n    while queue:\n        node, depth = queue.pop(0)\n        max_depth = max(max_depth, depth)\n        if node.left: queue.append((node.left, depth + 1))\n        if node.right: queue.append((node.right, depth + 1))\n    return max_depth",
        ],
    },
    {
        "name": "typescript_types",
        "positive": [
            "interface User {\n  id: number;\n  name: string;\n  email: string;\n}\n\nfunction getUser(id: number): User | undefined {\n  const users: User[] = [];\n  return users.find(u => u.id === id);\n}",
            "type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };\n\nfunction parseJson<T>(input: string): Result<T, Error> {\n  try {\n    return { ok: true, value: JSON.parse(input) };\n  } catch (e) {\n    return { ok: false, error: e as Error };\n  }\n}",
        ],
        "negative": [
            "function getUser(id) {\n  const users = [];\n  return users.find(u => u.id === id);\n}",
            "function parseJson(input) {\n  try {\n    return { ok: true, value: JSON.parse(input) };\n  } catch (e) {\n    return { ok: false, error: e };\n  }\n}",
        ],
    },
    {
        "name": "verbose_documentation",
        "positive": [
            "# This function performs a binary search on a sorted array.\n# It takes an array and a target value as input.\n# Returns the index of the target if found, otherwise -1.\ndef binary_search(arr, target):\n    # Initialize the low and high pointers\n    low = 0  # Start of the array\n    high = len(arr) - 1  # End of the array\n    # Continue searching while the search space is valid\n    while low <= high:\n        mid = (low + high) // 2  # Calculate the middle index\n        if arr[mid] == target:  # Found the target\n            return mid",
        ],
        "negative": [
            "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
        ],
    },
]
```

### How to compute feature activations for a snippet

For each snippet, tokenize it (`prepend_bos=False`), truncate to 512 tokens, run through the model, mean-pool the residual stream across the sequence dimension, then encode through the SAE:

```python
tokens = model.to_tokens(snippet, prepend_bos=False)
if tokens.shape[1] > 512:
    tokens = tokens[:, :512]

with torch.no_grad():
    _, cache = model.run_with_cache(tokens, names_filter=hook_point)

acts = cache[hook_point]                            # [1, seq_len, d_model]
mean_acts = acts[0].mean(dim=0).float().unsqueeze(0)  # [1, d_model]
feat_acts = sae.encode(mean_acts).squeeze(0)          # [d_sae]
```

This produces one activation vector per snippet. `feat_acts[i]` is a scalar: the activation of feature `i` for this snippet.

### Differential scoring

For each property, stack all positive snippet activations into a matrix `pos_acts` of shape `[n_pos, d_sae]`, and all negative snippet activations into `neg_acts` of shape `[n_neg, d_sae]`. Then:

```python
pos_mean = pos_acts.mean(dim=0)                        # [d_sae]
neg_mean = neg_acts.mean(dim=0)                        # [d_sae]
pos_freq = (pos_acts > 0).float().mean(dim=0)          # [d_sae]
neg_freq = (neg_acts > 0).float().mean(dim=0)          # [d_sae]

diff = pos_mean - neg_mean
score = diff * (pos_freq - neg_freq + 0.1)
```

`score[i]` is the differential score for feature `i` on this property. The `+0.1` bias term prevents zeroing out features where positive and negative frequencies are equal but mean activations differ.

Select the **top 10 features** per property by `score` (descending). Record for the top-1 feature:

- `top1_feature`: the feature index
- `top1_score`: `score[feature_idx]`
- `top1_specificity`: `pos_mean[feature_idx] / (neg_mean[feature_idx] + 1e-8)`
- `top1_pos_freq`: `pos_freq[feature_idx]`

Also compute `top5_mean_score`: mean of `score` for the top 5 features.

### Cross-property contamination

After computing top-5 feature indices for all 6 properties (30 feature indices total, potentially with repeats), count how many feature indices appear in the top-5 of **more than one** property. Those are contaminated — they respond to general code complexity rather than a specific property.

```python
from collections import Counter

all_top5 = []
for prop in properties:
    all_top5.extend(top5_indices_for[prop])

counts = Counter(all_top5)
contaminated = sum(1 for fid, c in counts.items() if c > 1)
total_unique = len(set(all_top5))
contamination_fraction = contaminated / max(total_unique, 1)
```

### Discovery score formula

```python
per_property_score = mean([top1_score for each property])
specificity_bonus = mean([top1_specificity for each property])
contamination_penalty = 1.0 - contamination_fraction

discovery_score = per_property_score * specificity_bonus * contamination_penalty
```

---

## Level 3: Steering Verification

Test whether the top-1 feature per property from Level 2 actually steers the model's generation toward that property when its decoder vector is added to the residual stream.

### Steering mechanism

Extract the feature's decoder vector from the SAE weight matrix and add it (scaled by a strength multiplier) to the residual stream at every token position during generation:

```python
steering_vector = sae.W_dec[feature_idx].detach().clone()

def steering_hook(value, hook):
    value[:, :, :] = value + strength * steering_vector.to(value.device, value.dtype)
    return value

tokens = model.to_tokens(prompt, prepend_bos=False)
with torch.no_grad():
    with model.hooks(fwd_hooks=[(hook_point, steering_hook)]):
        output_tokens = model.generate(
            tokens,
            max_new_tokens=200,
            temperature=0.0,
            do_sample=False,
        )

output = model.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
```

Use `temperature=0.0` and `do_sample=False` for deterministic outputs. At `strength=0.0`, generate without the hook (baseline).

### Strength sweep

Test 7 strengths per feature: `[-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]`

### Prompts

Use these ambiguous prompts that don't explicitly request any property, so any property appearing in the output is caused by the steering:

```python
STEERING_PROMPTS = [
    "Write a Python function that takes a list and returns the largest element",
    "Implement a stack data structure with push, pop, and peek methods",
    "Write a function that merges two sorted arrays into one sorted array",
    "Create a function that validates an email address",
]
```

### Property density measurement

For each generated output, count regex pattern matches to measure how much of the target property is present. Each property has its own pattern set:

```python
import re

DENSITY_PATTERNS = {
    "type_annotations": [
        r":\s*(?:int|float|str|bool|list|dict|tuple|set|None|Any|Optional|Union|Callable)\b",
        r"->\s*(?:int|float|str|bool|list|dict|tuple|set|None|Any|Optional|Union)\b",
        r":\s*(?:List|Dict|Tuple|Set)\[",
        r":\s*(?:number|string|boolean|void|never|any|unknown)\b",
        r":\s*(?:Array|Map|Set|Record|Promise|Partial)\s*<",
        r"\binterface\s+\w+",
        r"\btype\s+\w+\s*=",
    ],
    "error_handling": [
        r"\btry\b",
        r"\bexcept\b",
        r"\bcatch\b",
        r"\bfinally\b",
        r"\braise\b",
        r"\bthrow\b",
    ],
    "functional_style": [
        r"\bmap\s*\(",
        r"\bfilter\s*\(",
        r"\breduce\s*\(",
        r"\blambda\b",
        r"\.\bmap\b",
        r"\.\bfilter\b",
        r"\.\breduce\b",
        r"\[.+\bfor\b.+\bin\b.+\]",
    ],
    "recursion": [
        r"\breturn\s+\w+\s*\(",
        r"\brecurs",
    ],
    "typescript_types": [  # same patterns as type_annotations
        r":\s*(?:int|float|str|bool|list|dict|tuple|set|None|Any|Optional|Union|Callable)\b",
        r"->\s*(?:int|float|str|bool|list|dict|tuple|set|None|Any|Optional|Union)\b",
        r":\s*(?:number|string|boolean|void|never|any|unknown)\b",
        r":\s*(?:Array|Map|Set|Record|Promise|Partial)\s*<",
        r"\binterface\s+\w+",
        r"\btype\s+\w+\s*=",
    ],
    "verbose_documentation": [
        r'"""',
        r"'''",
        r"#\s+\S",
        r"//\s+\S",
        r"/\*",
        r"\bArgs:\b",
        r"\bReturns:\b",
        r"\bExample",
    ],
}

def compute_density(text, property_name):
    patterns = DENSITY_PATTERNS[property_name]
    lines = text.strip().split("\n")
    total_lines = max(len(lines), 1)
    total_matches = sum(
        len(re.findall(p, text, re.MULTILINE | re.IGNORECASE))
        for p in patterns
    )
    return total_matches / total_lines
```

### Monotonicity test

For each property's top-1 feature, compute density at each of the 7 strengths, averaged across all 4 prompts. Then:

```python
neg_avg = mean(densities at strengths < 0)    # strengths -5, -3, -1
baseline = density at strength 0.0
pos_avg = mean(densities at strengths > 0)    # strengths +1, +3, +5

is_monotonic = (pos_avg > baseline) and (baseline > neg_avg)
```

A feature passes if `is_monotonic is True`.

### Coherence check

At `strength=+5.0`, verify the output hasn't degenerated into gibberish. The output is coherent if it contains at least one recognizable code keyword:

```python
CODE_KEYWORDS = r"\b(def|function|class|return|if|for|while|import|const|let|var)\b"
is_coherent = bool(re.search(CODE_KEYWORDS, output_at_plus_5))
```

### Steering score formula

```python
property_scores = []
for each property:
    monotonic = 1 if is_monotonic else 0
    coherent  = 1 if is_coherent else 0
    effect_size = pos_avg - neg_avg

    property_scores.append(monotonic * coherent * effect_size)

steering_score = mean(property_scores)
```

---

## Final Ranking

Combine the three levels:

```
final_score = health_score * discovery_score * (1 + steering_score)
```

- An SAE with 90% dead features gets a low `health_score`, dragging down everything.
- An SAE with no property-specific features gets a low `discovery_score`, dragging down everything.
- `(1 + steering_score)` means steering is a bonus multiplier: an SAE with no monotonic features gets 1x (no penalty, no reward); an SAE with strong monotonic steering gets >1x.

Sort SAEs by `final_score` descending. The first entry is the best SAE.

---

## Output

### JSON file

Write results to `sae_evaluation_results.json`:

```json
{
  "ranking": [
    {
      "rank": 1,
      "name": "custom_v2",
      "final_score": 1.847,
      "health": {
        "health_score": 0.93,
        "d_sae": 32768,
        "dead_fraction": 0.07,
        "alive_count": 30474,
        "l0_mean": 63.8,
        "l0_std": 12.4,
        "decoder_norm_mean": 0.251,
        "decoder_norm_std": 0.041
      },
      "discovery": {
        "discovery_score": 2.15,
        "contamination_fraction": 0.0,
        "per_property": {
          "type_annotations": {
            "top1_feature": 4521,
            "top1_score": 3.42,
            "top1_specificity": 18.7,
            "top1_pos_freq": 1.0,
            "top5_mean_score": 2.14
          },
          "error_handling": { "..." : "..." },
          "functional_style": { "..." : "..." },
          "recursion": { "..." : "..." },
          "typescript_types": { "..." : "..." },
          "verbose_documentation": { "..." : "..." }
        }
      },
      "steering": {
        "steering_score": 0.42,
        "per_property": {
          "type_annotations": {
            "feature": 4521,
            "is_monotonic": true,
            "is_coherent": true,
            "neg_avg_density": 0.0,
            "baseline_density": 0.02,
            "pos_avg_density": 0.35,
            "effect_size": 0.35
          },
          "error_handling": { "..." : "..." },
          "functional_style": { "..." : "..." },
          "recursion": { "..." : "..." },
          "typescript_types": { "..." : "..." },
          "verbose_documentation": { "..." : "..." }
        }
      }
    }
  ]
}
```

### Printed summary table

Print a table to stdout:

```
==============================================================================
SAE Comparative Evaluation Results
==============================================================================
Rank  Name          Final    Health   Discovery  Steering  Dead%   L0 mean
------------------------------------------------------------------------------
  1   custom_v2     1.847    0.930    2.150      0.420      7.0%    63.8
  2   custom_v4     1.623    0.910    1.940      0.310      9.0%    61.2
  3   custom_v1     1.105    0.870    1.380      0.180     13.0%    58.4
  4   custom_v3     0.742    0.850    0.940      0.070     15.0%    55.1
  5   community     0.098    0.200    0.520      0.000     80.0%   128.0
==============================================================================

Monotonic steering achieved:
  custom_v2:  type_annotations, error_handling, recursion (3/6)
  custom_v4:  error_handling, functional_style (2/6)
  custom_v1:  recursion (1/6)
  custom_v3:  (0/6)
  community:  (0/6)

Winner: custom_v2
```

---

## CLI Interface

```
usage: evaluate_saes.py [-h] --sae-paths SAE_PATHS [SAE_PATHS ...]
                        --sae-names SAE_NAMES [SAE_NAMES ...]
                        [--sae-pretrained SAE_PRETRAINED [SAE_PRETRAINED ...]]
                        [--device DEVICE]
                        [--output OUTPUT]

  --sae-paths         Paths/releases for each SAE (space-separated)
  --sae-names         Human-readable names (space-separated, same order)
  --sae-pretrained    Indices (0-based) of SAEs that are HuggingFace-hosted
                      (use from_pretrained instead of load_from_disk)
  --device            GPU device (default: cuda:0)
  --output            Output JSON path (default: sae_evaluation_results.json)
```

Example:

```bash
python evaluate_saes.py \
    --sae-paths \
        tylercosgrove/mistral-7b-sparse-autoencoder-layer16 \
        ~/checkpoints/v1/final \
        ~/checkpoints/v2/final \
        ~/checkpoints/v3/final \
        ~/checkpoints/v4/final \
    --sae-names community custom_v1 custom_v2 custom_v3 custom_v4 \
    --sae-pretrained 0 \
    --device cuda:0 \
    --output sae_evaluation_results.json
```
