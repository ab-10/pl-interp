# Quick Feature Discovery Experiment

Find SAE features that produce visible steering effects in the web UI, ready for
a hackathon demo. Run time: ~30–45 minutes on one H100.

## Deliverable

A JSON file containing:

```json
{
  "features": [
    {
      "layer": 18,
      "feature_idx": 304,
      "label": "Type annotations",
      "demo_strength": 7.0,
      "direction": "positive amplifies, negative suppresses"
    }
  ],
  "demo_prompt": "Write a Python function that merges two sorted lists.",
  "temperature": 0.3
}
```

This maps directly into `FEATURE_LABELS` in `backend/server.py` and the slider
settings in the web UI (range -10 to +10, step 0.5).

---

## What to Run

One script, three phases: generate, rank, verify. All on the GPU VM.

### Phase 1: Generate (15 min)

For each of 5 code properties, generate 5 target and 5 control completions
using the unmodified model. Target and control prompts differ in exactly one
axis.

**Properties and prompt pairs:**

**Type annotations**
| Target | Control |
|--------|---------|
| Write a Python function with full type annotations that merges two sorted lists | Write a Python function that merges two sorted lists |
| Write a TypeScript function with explicit types that parses a CSV string into rows | Write a function that parses a CSV string into rows |
| Write a Python function with type hints for parameters and return value that filters a dictionary by value | Write a Python function that filters a dictionary by value |
| Write a typed Python function that converts a nested JSON object to a flat dictionary | Write a Python function that converts a nested JSON object to a flat dictionary |
| Write a TypeScript function with generic type parameters that removes duplicates from an array | Write a function that removes duplicates from an array |

**Error handling**
| Target | Control |
|--------|---------|
| Write a Python function that reads a JSON file with try/except for FileNotFoundError and JSONDecodeError | Write a Python function that reads a JSON file |
| Write a function that connects to an API with error handling for timeouts and HTTP errors | Write a function that connects to an API |
| Write a Python function with comprehensive error handling that parses user input as a number | Write a Python function that parses user input as a number |
| Write a function that opens a database connection with try/except and retry logic | Write a function that opens a database connection |
| Write a function with error handling that writes data to a CSV file | Write a function that writes data to a CSV file |

**Recursion**
| Target | Control |
|--------|---------|
| Write a recursive Python function that flattens a nested list | Write a Python function that flattens a nested list |
| Write a recursive function that computes the nth Fibonacci number | Write a function that computes the nth Fibonacci number using a loop |
| Write a recursive function to find all permutations of a string | Write a function to find all permutations of a string |
| Write a recursive depth-first search on an adjacency list graph | Write a breadth-first search on an adjacency list graph |
| Write a recursive function that computes the power set of a list | Write a function that computes the power set of a list |

**Verbose comments**
| Target | Control |
|--------|---------|
| Write a well-documented Python function with a detailed docstring and inline comments that implements binary search | Write a Python binary search function with no comments |
| Write a sorting function with a comment on every line explaining the logic | Write a sorting function |
| Write a Python class with comprehensive docstrings on every method that implements a stack | Write a Python stack class |
| Write a heavily commented function that validates an email address | Write a function that validates an email address |
| Write a function with detailed inline documentation that parses command-line arguments | Write a function that parses command-line arguments |

**Functional style**
| Target | Control |
|--------|---------|
| Write Python code using map, filter, and reduce to process a list of numbers | Write Python code using for loops to process a list of numbers |
| Write a data pipeline using only pure functions and function composition | Write a data pipeline using a class with mutable state |
| Write TypeScript using Array.map and filter to transform objects | Write TypeScript using for loops to transform objects |
| Write Python using list comprehensions and generator expressions instead of loops | Write Python using for loops with append |
| Write a validation pipeline using higher-order functions | Write a validation function using if/else chains |

**For each prompt:**

1. Tokenize and generate 300 tokens at temperature 0.3
2. Run the full output back through the model (forward pass)
3. Capture residual stream at layers 18 and 27 via PyTorch hooks
4. Encode each token position through the corresponding SAE individually
5. Record per-feature firing frequency and mean activation over the
   generation tokens only (exclude the prompt prefix)

```python
captured = {}

def make_capture_hook(name):
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured[name] = hidden.detach()
    return hook

handles = [
    model.model.layers[18].register_forward_hook(make_capture_hook("L18")),
    model.model.layers[27].register_forward_hook(make_capture_hook("L27")),
]

with torch.no_grad():
    model(output_ids)

for h in handles:
    h.remove()

# Per-token encode (NOT mean-then-encode)
for layer_key, sae in [("L18", sae_l18), ("L27", sae_l27)]:
    acts = captured[layer_key][0, prompt_len:]  # generation tokens only
    feats = sae.encode(acts.float())             # [gen_len, d_sae]
    firing = (feats > 0).float()
    freq = firing.mean(dim=0)                    # [d_sae]
    mean_act = feats.sum(dim=0) / (firing.sum(dim=0) + 1e-8)
```

### Phase 2: Rank (< 1 min)

For each property and each layer, compute a differential score per feature:

```
diff = target_mean_freq * target_mean_act − control_mean_freq * control_mean_act
```

Filter to features where:
- `diff > 0`
- Target cross-prompt frequency >= 0.4 (fires in at least 2 of 5 target prompts)

Take the top 5 features per layer per property. This gives at most
5 properties x 2 layers x 5 = 50 candidate features.

### Phase 3: Verify (15 min)

For each candidate feature, steer generation on 3 neutral prompts and check
whether the output visibly changes in the expected direction.

**Neutral verification prompts:**
1. `Write a Python function that merges two sorted lists.`
2. `Implement a function that checks if a string is a palindrome.`
3. `Write a function that counts word frequencies in a string.`

**Steering setup:** Apply the decoder direction at the feature's own layer
during generation tokens only. Test strengths 3.0, 5.0, 8.0.

```python
def make_steering_hook(sae, feature_idx, strength):
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] > 1:           # prompt prefill — skip
            return output
        vec = sae.W_dec[feature_idx]
        hidden += strength * vec.to(hidden.device, hidden.dtype)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook

handle = model.model.layers[LAYER].register_forward_hook(
    make_steering_hook(sae, feat_idx, strength)
)
steered = model.generate(**inputs, max_new_tokens=200, temperature=0.3, do_sample=True)
handle.remove()
```

**Pass criteria** (eyeball check, no automated metric):
- At strength +5.0, the target property is clearly present in the output
  when it was absent or weak in the baseline
- At strength -5.0, the target property is reduced or absent
- The output is coherent (no degenerate loops or gibberish)
- The effect is consistent across at least 2 of 3 neutral prompts

Drop any feature that fails. For features that pass, record the strength that
produces a clear effect without breaking coherence — this becomes `demo_strength`
in the output JSON.

### Output

Write `results/demo_features.json`:

```json
{
  "features": [
    {
      "layer": 18,
      "feature_idx": 304,
      "label": "Type annotations",
      "demo_strength": 7.0,
      "neg_strength": -5.0,
      "verified_on": ["merge sorted lists", "palindrome", "word freq"]
    }
  ],
  "demo_prompt": "Write a Python function that merges two sorted lists.",
  "backup_prompts": [
    "Implement a function that checks if a string is a palindrome.",
    "Write a function that counts word frequencies in a string."
  ],
  "settings": {
    "temperature": 0.3,
    "max_new_tokens": 200,
    "slider_range": [-10, 10],
    "slider_step": 0.5
  }
}
```

Then update `backend/server.py` `FEATURE_LABELS`:

```python
FEATURE_LABELS: dict[int, dict[int, str]] = {
    18: {
        304: "Type annotations",
        # ... other verified layer-18 features
    },
    27: {
        1052: "Verbose comments & documentation",
        # ... other verified layer-27 features
    },
}
```

---

## Demo Runbook

After the experiment:

1. SSH to VM, update `FEATURE_LABELS` in `server.py` with discovered features
2. Start backend: `uvicorn backend.server:app --host 0.0.0.0 --port 8000`
3. Locally: `npm run dev` (Next.js on port 3000, proxies to VM via SSH tunnel)
4. Open web UI, verify sliders appear for each discovered feature
5. Enter the demo prompt: `Write a Python function that merges two sorted lists.`
6. Generate baseline (all sliders at 0)
7. Crank one slider to its `demo_strength`, generate again, show the diff

**Good demo sequence:**
- Start with all sliders at 0, generate baseline
- Push "Type annotations" to +7, regenerate — output gains type hints
- Push "Type annotations" to -5, regenerate — output loses any type hints
- Reset, push "Verbose comments" to +8 — output gains docstrings and inline comments
- Push both "Type annotations" +7 and "Verbose comments" +8 — output gains both

This shows single-axis steering, suppression, and multi-feature composition.
