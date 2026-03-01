# Bugs in Feature Discovery Pipeline

This document catalogs bugs and correctness issues identified in the current SAE
feature discovery and steering code by comparing the implementation against the
method described in [Tyler Cosgrove's blog post](https://www.tylercosgrove.com/blog/exploring-sae/)
and [reference repository](https://github.com/tylercosgrove/sparse-autoencoder-mistral7b).

---

## Bug 1: Encoding the Mean Instead of Meaning the Encodings

**Severity:** High — produces mathematically incorrect feature rankings
**Files affected:**
- `scripts/05_prompt_and_observe.py` lines 168–170
- `scripts/explore_trained_sae.py` lines 111–113

**The bug:**

Both scripts mean-pool the residual stream activations *before* passing them
through the SAE encoder:

```python
# 05_prompt_and_observe.py:168-170
mean_acts = acts[0].mean(dim=0).float().unsqueeze(0)  # [1, d_model]
mean_feat_acts = sae.encode(mean_acts).squeeze(0)      # [d_sae]

# explore_trained_sae.py:111-113
mean_acts = acts.mean(dim=1)                            # [1, d_model]
feat_acts = sae.encode(mean_acts.to(torch.float32)).squeeze(0)
```

The SAE encoder is a nonlinear function. It applies a linear transformation
(`x @ W_enc + b_enc`) followed by top-k activation, which zeros out all but the
k largest values. For any nonlinear function f:

    f(mean(x_1, ..., x_n)) ≠ mean(f(x_1), ..., f(x_n))

Concretely: a feature that fires strongly at 10 out of 200 token positions gets
diluted to near-zero when its signal is averaged in residual space before top-k
selection. The averaged residual vector is dominated by features that fire weakly
but everywhere, not features that fire strongly at specific positions. After
top-k, the strong-but-sparse feature may be completely zeroed out.

**The fix:**

Encode each token position individually, then aggregate in feature space:

```python
all_feat_acts = sae.encode(acts[0].float())  # [seq_len, d_sae]
mean_feat_acts = all_feat_acts.mean(dim=0)    # [d_sae]
```

Note that `explore_trained_sae.py` already does per-token encoding correctly in
its `profile_sae()` function (lines 159–161) for computing L0 sparsity, but
then uses the incorrect mean-then-encode approach in `collect_activations()` for
the contrastive analysis that actually discovers features.

---

## Bug 2: Last-Token-Only Activation Collection

**Severity:** Medium — discards the majority of available signal
**File:** `scripts/01_collect_activations.py` lines 145–150

**The bug:**

The main activation collection pipeline extracts SAE features from only the
final token position of each prompt:

```python
# 01_collect_activations.py:146-150
acts = cache[hook_point]          # [1, seq_len, d_model]
last_token_acts = acts[0, -1, :]  # [d_model]  <-- discards positions 0 to seq_len-2
feature_acts = sae.encode(last_token_acts.float().unsqueeze(0))
```

For a 200-token code prompt, this throws away activations from 199 of 200
positions. The last token's features reflect what the model predicts *next*,
not what features the SAE associates with the code *content*. A Python function
definition ending with `return result` will produce features related to
"newline/end-of-function" rather than features related to recursion, type
annotations, or error handling patterns present throughout the function body.

This is the approach Cosgrove's blog describes as Method 1 — record a small
number of top tokens/activations per feature — which the blog explicitly calls
out as producing "largely unintelligible results."

**The fix:**

Encode every token position through the SAE and aggregate feature activations
across the full sequence, as described in Bug 1's fix. Optionally also track
per-position firing frequency:

```python
acts = cache[hook_point]                         # [1, seq_len, d_model]
all_feat_acts = sae.encode(acts[0].float())      # [seq_len, d_sae]
mean_feat_acts = all_feat_acts.mean(dim=0)        # [d_sae]
firing_freq = (all_feat_acts > 0).float().mean(dim=0)  # fraction of positions
```

---

## Bug 3: Steering Applied During Prompt Encoding, Skipped During Generation

**Severity:** Medium — inverts the intended steering behavior
**Files affected:**
- `backend/server.py` lines 112–117
- `scripts/explore_trained_sae.py` lines 322–323, 332–333

**The bug:**

The steering hook skips single-token forward passes and applies to multi-token
passes:

```python
# backend/server.py:112-117
def steering_hook(value, hook):
    if value.shape[1] == 1:
        return value                 # <-- skips generation tokens
    for strength, vec in steering_vectors:
        value[:, :, :] = value + strength * vec.to(value.device, value.dtype)
    return value                     # <-- steers prompt encoding
```

In transformer_lens's `model.generate()` with KV caching:
- The first forward pass processes all prompt tokens at once: shape
  `[1, prompt_len, d_model]` — steering IS applied
- Each subsequent generation step processes one new token: shape
  `[1, 1, d_model]` — steering is NOT applied

This means the steering vector modifies the model's representation of the input
prompt but does not steer the generation of new tokens. Generation is only
influenced indirectly, through attention to the modified prompt KV cache.

Cosgrove's reference implementation in `utils/generate.py` does the opposite —
SAE intervention is applied only during generation, not during prompt encoding:

```python
# From Cosgrove's utils/generate.py
is_generating = i >= max_prompt_len
using_sae = is_generating   # SAE only during generation, NOT prompt
```

The rationale: steering during prompt encoding distorts the model's
comprehension of the user's input. Steering during generation directly biases
what the model produces at each step.

The current code follows the pattern from Cosgrove's simpler `demo.py` rather
than the more correct `generate.py`. For a web application where users enter
diverse prompts and expect the model to understand them before generating
steered output, prompt-phase-only steering is the wrong choice.

**The fix:**

Invert the condition so steering applies during generation steps and is skipped
during prompt encoding:

```python
def steering_hook(value, hook):
    if value.shape[1] > 1:
        return value                 # skip prompt encoding
    for strength, vec in steering_vectors:
        value += strength * vec.to(value.device, value.dtype)
    return value
```

Or apply to both phases if the goal is maximum steering effect, as some
applications in the Anthropic steering literature do.

---

## Bug 4: Single-Token Prompts Receive No Steering

**Severity:** Low — edge case
**File:** `backend/server.py` line 113

**The bug:**

The `shape[1] == 1` guard applies to any forward pass with a single token,
including the initial prompt encoding when the user submits a one-token prompt.
In that case, both the prompt encoding and all generation steps have
`shape[1] == 1`, so the steering hook never fires and generation is completely
unsteered.

**The fix:**

Track whether the first (prompt) pass has occurred, rather than relying on
tensor shape:

```python
seen_prompt = False

def steering_hook(value, hook):
    nonlocal seen_prompt
    if not seen_prompt:
        seen_prompt = True
        return value  # skip prompt
    # apply steering to generation tokens
    for strength, vec in steering_vectors:
        value += strength * vec.to(value.device, value.dtype)
    return value
```

---

## Bug 5: No Generation Step in the Main Collection Pipeline

**Severity:** Medium — uses a weaker method when a better one is available
**File:** `scripts/01_collect_activations.py`

**The bug:**

Script 01 runs each prompt through the model as a forward pass only — it never
generates a completion. It records which features fire when the model *reads*
code, not which features the model *uses* when it writes code.

Cosgrove's blog describes this as Method 2 (positive/negative dataset
weighting) and identifies its key limitation: features found this way tend to be
overly broad. Searching for specific code properties (e.g., recursion) returns
features that correspond to "code in general" rather than the target property,
because the contrastive sets (code vs. non-code) differ along too many
dimensions simultaneously.

The blog's Method 3 — prompt the model to generate relevant output, then record
which features score highest across the generation — avoids this problem because
the model's own generation process selectively activates the features it uses to
produce the target property.

Script 05 (`prompt_and_observe.py`) already implements generation-based
collection, but the main pipeline (scripts 01–02) does not. The feature
registry in `backend/server.py` was built from the contrastive (non-generation)
pipeline, meaning the discovered features may be "code in general" features
rather than the specific property features they are labeled as.

**The fix:**

See the companion document `GENERATION_BASED_DISCOVERY.md` for a complete
generation-based approach.
