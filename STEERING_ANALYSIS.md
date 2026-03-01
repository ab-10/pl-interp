# Steering Analysis Results

## Summary

The backend had three bugs (now fixed) that degraded steering quality, but the primary reason steering doesn't produce visible typing differences is that **the discovered features don't control type annotations**. They shift language identity (toward JS/TS) and output verbosity, but produce zero type annotation markers at any steering strength tested, up to +500.

---

## 1. SAE Profile

Characterization of `tylercosgrove/mistral-7b-sparse-autoencoder-layer16` via `scripts/00_explore_sae.py`:

| Property | Value | Implication |
|---|---|---|
| Type | TopK (K=128) | Every token fires exactly 128 features, zero variance |
| Dictionary size | 131,072 features | Large dictionary but mostly unused |
| Dead features | 104,821 (80%) | Only 26,251 features ever fire on diverse prompts |
| Decoder norm range | 0.003 – 0.658 | 237x variation; strength=3.0 means different things per feature |
| Decoder norm median | 0.182 | Most features have small decoder vectors |

### Perturbation scale

MLP output at layer 16 has norms of ~2.4–3.4. Steering at strength=3.0 with a typical feature (decoder norm ~0.25) produces a perturbation of ~0.76, which is **22–31% of the residual norm** — a non-trivial perturbation that should have measurable effects.

| Strength | Perturbation (norm 0.25 feature) | % of residual norm |
|---|---|---|
| 3.0 | 0.76 | ~25% |
| 10.0 | 2.55 | ~85% |
| 20.0 | 5.10 | ~170% |
| 50.0 | 12.75 | ~425% |

---

## 2. Backend Bug Fixes Applied

### Bug 1: Missing `dtype=torch.float16` on model loading

Model was loading in default dtype (float32), using ~32 GB VRAM. Now loads in float16 at 14.7 GB, matching the SAE's dtype and all reference scripts.

### Bug 2: Low-quality steering hook

Old hook held live references to SAE parameters and risked dtype mismatches:
```python
# Old (broken)
def steering_hook(value, hook):
    for feat_id, strength in active:
        value = value + strength * sae.W_dec[feat_id]
    return value
```

Fixed hook pre-computes detached vectors with dtype conversion:
```python
# New (fixed)
steering_vectors = []
for feat_id, strength in active:
    steering_vectors.append((strength, sae.W_dec[feat_id].detach().clone()))

def steering_hook(value, hook):
    for strength, vec in steering_vectors:
        value[:, :, :] = value + strength * vec.to(value.device, value.dtype)
    return value
```

### Bug 3: Hardcoded `temperature=0`

Greedy decoding masked steering effects — if the perturbation doesn't change the argmax token, the output is identical. Now defaults to `temperature=0.3` with `do_sample=True`, exposed as a frontend slider.

### Impact of fixes

The fixes are correct and necessary, but they don't resolve the core problem. With temperature=0.3, outputs vary between runs, but no type annotations appear at any strength.

---

## 3. Feature Discovery: Original Pipeline

The original 20-feature registry (`results/feature_registry.json`) contains only "code stub format" detectors. Every label describes the HumanEval/MultiPL-E format, not a code property:

| Feature | Label |
|---|---|
| 17411 | R-style function stubs with doctests |
| 8081 | C++ Competitive Programming Stubs |
| 59751 | Rust function docstring examples |
| 32104 | C++ Function Docstring Examples |
| 18872 | C++ Kata Problem Descriptions with Examples |
| 40901 | Incomplete or Malformed Coding Problems |
| 53942 | Math/CS Problem Statement Detection |
| ... | (all similar) |

**Root cause**: Homogeneous HumanEval stubs as the code dataset + code-vs-prose contrast finds "is this code?" features, not "what kind of code?" features.

---

## 4. Feature Discovery: Typing Experiment

The `scripts/typing/` contrastive pipeline found 20 features labeled "Explicit type annotations present" by the Mistral API. Two (124809, 6133) were cross-language consensus features.

**Critical finding: `04_verify_steering.py` was never run before the features were labeled as "[verified]" in the web UI.** The verification file (`steering_verification.json`) did not exist in `scripts/typing/results/`.

### Verification results (now run)

Typing density (type annotation markers per line) across steering strengths for the top 3 features, averaged over 3 prompts:

| Strength | Feature 124809 | Feature 6133 | Feature 8019 |
|---|---|---|---|
| -5.0 | 0.000 | 0.000 | 0.000 |
| -3.0 | 0.028 | 0.000 | 0.000 |
| -1.0 | 0.000 | 0.000 | 0.011 |
| 0.0 (baseline) | 0.000 | 0.000 | 0.000 |
| +1.0 | 0.000 | 0.000 | 0.000 |
| +3.0 | 0.000 | 0.000 | 0.000 |
| +5.0 | 0.000 | 0.000 | 0.000 |

**Monotonic trend: None.** No feature shows increasing type annotation density with increasing strength. The non-zero values at -3.0 and -1.0 are noise (one marker in one prompt).

---

## 5. Direct Steering Tests via Backend

### Feature 124809 ("type annotations") at various strengths

| Strength | Type markers | Observed effect |
|---|---|---|
| +5 | 0 | Slightly different code, no types |
| +10 | 0 | More comments added, no types |
| +20 | 0 | Similar code structure, no types |
| +50 | 0 | Similar code structure, no types |
| +100 | 0 | Similar code structure, no types |
| +200 | 0 | Similar code structure, no types |
| +500 | 0 | Gibberish ("megorao ojo"), still no types |
| -10 | 0 | Dramatically terse output ("A: max(lst)") |
| -20 | 0 | Extremely terse output |
| -50 | 0 | Extremely terse ("A: max(my_list)") |

### All 6 typing features stacked at +50

Prompt: "Implement a binary search function"

- **Baseline**: Python implementation with explanation
- **+50 (all features)**: Switched to **JavaScript** (`function binarySearch(arr, target) { ... }`), but still zero TypeScript type annotations
- **-50 (all features)**: Python implementation, similar to baseline

**The features shift language identity (Python → JS/TS) rather than adding type annotations.**

### Error handling feature 20704 at +20

No try/except blocks added. Output nearly identical to baseline.

---

## 6. Prompt-and-Observe Results

New property-specific features discovered via `scripts/05_prompt_and_observe.py`:

### Cross-property contamination

Feature 74449 appears in the top 20 for ALL four properties tested:

| Property | Rank | Target Freq | Target Act | Control Freq | Control Act |
|---|---|---|---|---|---|
| error_handling | 4 | 100% | 23.6 | 80% | 23.3 |
| recursion | 1 | 80% | 17.1 | 60% | 12.9 |
| functional_style | 6 | 100% | 30.6 | 100% | 27.0 |
| verbosity | 1 | 100% | 31.1 | 60% | 16.1 |

This is a generic "code complexity/elaboration" detector, not a property-specific feature.

### Property-exclusive features

Despite the contamination, 18–19 out of 20 top features per property are exclusive to that property:

| Property | Exclusive / Total |
|---|---|
| error_handling | 18/20 |
| recursion | 19/20 |
| functional_style | 19/20 |
| verbosity | 18/20 |

Top candidates per property (exclusive, highest differential score):

| Property | Feature | Diff Score | Target Freq | Control Freq |
|---|---|---|---|---|
| error_handling | 20704 | 16.03 | 80% | 0% |
| error_handling | 58573 | 8.63 | 20% | 0% |
| recursion | 83846 | 2.57 | 40% | 0% |
| functional_style | 46874 | 7.60 | 60% | 0% |
| verbosity | 73781 | 8.02 | 40% | 0% |

These are statistically underpowered (N=5 prompts per side) and would need validation via the full contrastive pipeline before use.

---

## 7. Root Cause Analysis

### Why the typing features don't steer type annotations

**1. The SAE is low-quality for fine-grained property control.**
- 80% dead features indicate the SAE is undertrained
- TopK-128 forces exactly 128 features per token; features compete for slots rather than composing
- Community-trained on general data, not optimized for code property decomposition

**2. The contrastive methodology conflated language identity with typing.**
The typing experiment compared TypeScript (typed) vs JavaScript (types stripped). Features that fire more on TypeScript than JavaScript pick up ANY TypeScript signal — including the language itself — not specifically type annotations. Evidence: stacking all 6 features at +50 switches the model to JavaScript output, not to typed Python.

**3. Detection ≠ Generation.**
Features that fire when the model *reads* typed code don't necessarily cause the model to *write* typed code when their decoder vectors are added. The SAE captures input-side activation patterns; the causal direction needed for generation steering may be different or may not exist at this layer.

**4. The hook point (`blocks.16.hook_mlp_out`) may not be suitable for generation steering.**
Layer 16 is the midpoint of a 32-layer model. Steering the MLP output at the midpoint may be too early to directly influence token-level output decisions, which are determined in later layers.

### What the steering mechanism DOES do

The mechanism works — it's just not producing the labeled effect:
- **Positive steering**: Shifts toward shorter, code-centric output; shifts language toward JS/TS
- **Negative steering**: Produces dramatically terse output ("A: max(lst)")
- **Extreme positive (+500)**: Produces gibberish, confirming the vector IS influencing the residual stream

---

## 8. Decoder Norms for Key Features

| Feature | Label | Decoder Norm | Perturbation @ str=3 | Perturbation @ str=10 |
|---|---|---|---|---|
| 124809 | type annotations (consensus) | 0.2548 | 0.76 | 2.55 |
| 6133 | static typing (consensus) | 0.2434 | 0.73 | 2.43 |
| 8019 | TypeScript type annotations | 0.2401 | 0.72 | 2.40 |
| 28468 | explicit type signatures | 0.2593 | 0.78 | 2.59 |
| 95915 | generic type parameters | 0.2533 | 0.76 | 2.53 |
| 70728 | Python type hints | 0.2891 | 0.87 | 2.89 |
| 20704 | error handling (new) | 0.2402 | 0.72 | 2.40 |

All features have similar decoder norms (~0.24–0.29), so norm variation is not a factor here.

---

## 9. Recommendations

### To make steering work for this project

1. **Try a different SAE.** The 80% dead feature rate and TopK-128 constraint limit this SAE's ability to represent fine-grained code properties. Look for SAEs trained specifically on code models or with higher alive-feature rates.

2. **Steer at a later layer.** The current hook point is layer 16 of 32 (MLP output). Try layers 24–28, closer to the output, where interventions more directly affect token predictions.

3. **Use within-language contrasts.** To find genuine typing features, compare typed Python vs untyped Python (not TypeScript vs JavaScript). The `scripts/typing/00_generate_dataset.py` already generates these pairs — but the features found from them still need to be validated as generation-steering features, not just detection features.

4. **Validate with generation-time activations.** Instead of measuring activations on input code, measure which features fire when the model IS generating typed code (during the generation loop). Features that are active during generation of type annotations are more likely to steer generation.

5. **Try activation patching** instead of additive steering. Rather than adding a steering vector, clamp specific feature activations to high/low values during generation. This is more targeted than additive perturbation.

### To use the current features in the web UI

The current features are mislabeled. They should be relabeled to reflect their actual effect:
- Positive steering: shifts toward concise code output, JS/TS language family
- Negative steering: produces terse, minimal output

If keeping them, update labels to set accurate user expectations.
