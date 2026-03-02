# Steering Code Generation with Sparse Autoencoders

**Can we find individual neurons inside an LLM that control *how* it writes code — and then turn them up or down like sliders?**

We opened up Ministral 8B, trained custom Sparse Autoencoders on code, and found 30 verified features that let us toggle specific coding behaviors: type annotations, recursion, error handling, functional style, and documentation verbosity. Then we built a web UI to steer the model in real time.

## The Headline Result

Same prompt. Same model. Different feature activations:

```
Prompt: "Write a Python function that merges two sorted lists."
```

**Baseline** (no steering) — iterative two-pointer merge:
```python
def merge_sorted_lists(list1, list2):
    merged_list = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        ...
```

**Type Annotations feature ON** (+3.0) — adds `from typing import List`, full signatures:
```python
from typing import List

def merge_sorted_lists(list1: List[int], list2: List[int]) -> List[int]:
    merged_list: List[int] = []
    i: int = 0
    j: int = 0
    ...
```

**Recursion feature ON** (+3.0) — completely restructures the algorithm:
```python
def merge_sorted_lists(list1, list2):
    if not list1:
        return list2
    if not list2:
        return list1
    if list1[0] < list2[0]:
        return [list1[0]] + merge_sorted_lists(list1[1:], list2)
    else:
        return [list2[0]] + merge_sorted_lists(list1, list2[1:])
```

The model isn't being re-prompted — the same neural pathway is activated differently. We're reaching into the model's intermediate representations and amplifying the direction that encodes "use recursion" or "add type annotations."

---

## What We Learned About Mistral

### 1. Code properties are encoded as separable directions in activation space

The model doesn't store "type annotations" and "recursion" in the same tangled mess. At layer 18 (50% depth in Ministral 8B's 36-layer architecture), these properties decompose into distinct SAE features with minimal cross-contamination:

| Property | Best Feature | Specificity | Diff Score | Effect |
|---|---|---|---|---|
| **Type annotations** | L18:13176 | **1,499x** | 1.40 | Adds `List[int]`, `-> bool`, `from typing import` |
| **Functional style** | L18:16149 | **11.6x** | 2.13 | Switches to `map()`, `filter()`, `lambda` |
| **Recursive patterns** | L18:16290 | 1.5x | 1.61 | Rewrites iterative code as recursive |
| **Verbose comments** | L18:9344 | **5.6x** | 0.89 | Adds inline documentation to every block |
| **Error handling** | L18:9742 | 1.5x | 1.02 | Introduces `try`/`except` patterns |

Feature 13176 fires on 41% of tokens during typed code generation and on 0.03% during untyped code — a 1,499x specificity ratio. This isn't a noisy correlate. It's a genuine "typing" direction in the model's representation.

### 2. Community SAEs fail on domain-specific tasks — custom training is essential

We started with a widely-used community SAE for Mistral 7B (`tylercosgrove/mistral-7b-sparse-autoencoder-layer16`). It completely failed:

| | Community SAE | Our Custom SAE |
|---|---|---|
| Dead features | **80%** (104,821/131K) | 52% (17,125/32K) |
| Sparsity | Fixed 128 per token | Variable (mean 102, range 23-777) |
| Training data | Pile (general text) | **StarCoderData (code)** |
| Hook point | MLP output | **Residual stream** |
| Architecture | TopK | **BatchTopK** |
| Steering at +5 for typing | Zero type annotations | **Full typed signatures** |
| Steering at +500 for typing | Still zero. Then gibberish. | N/A — works at +3 |

The community SAE's "type annotation" features actually encoded *language identity* (Python vs TypeScript), not the presence of types. Boosting them switched the model to JavaScript — but still without any type annotations. Our code-trained SAE found the actual typing direction.

### 3. Detection and generation use different feature circuits

A key finding from our analysis: features that *detect* a property in input don't necessarily *cause* that property in output. The community SAE found features that fired when reading typed code, but adding their decoder vectors during generation had zero effect.

Our generation-based discovery method (the "Cosgrove Method 3") solved this by measuring activations *during generation*, not during reading. We generated typed and untyped code from the same prompts, then found features that differentially fired during the generation of typed tokens. These features steer generation because they're part of the generative circuit, not the comprehension circuit.

### 4. Multi-layer analysis reveals feature specialization by depth

We trained SAEs at both layer 18 (50% depth) and layer 27 (75% depth) of Ministral 8B:

- **Layer 18 features** encode *structural intent* — "use recursion," "add type annotations," "write functional code." These are planning-level representations.
- **Layer 27 features** encode *output refinement* — they overlap more across properties and act more like "how to format this output" than "what kind of code to write."

The sharpest features live at 50% depth. This aligns with the "semantic bottleneck" hypothesis — mid-network layers compress input into abstract representations before the model commits to specific output tokens.

### 5. Every prompt variant hurts correctness

We ran the full pipeline on HumanEval + MBPP with 6 prompt variants (baseline, typed, error_handling, decomposition, invariants, control_flow) across both Mistral 7B and Ministral 8B:

| Variant | Ministral 8B Pass Rate | Delta |
|---|---|---|
| **Baseline** | **36.8%** | — |
| Typed | 35.3% | -1.5 |
| Control flow | 34.1% | -2.7 |
| Error handling | 33.8% | -3.0 |
| Invariants | 33.9% | -2.9 |
| **Decomposition** | **28.6%** | **-8.2** |

Asking the model to add structure (types, error handling, decomposition) consistently *reduces* correctness. Decomposition — "use helper functions" — is the most harmful at -8pp. The model's correctness and structure are in tension, not alignment.

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │     Next.js Frontend         │
                    │  Feature sliders + diff view │
                    │  Activation visualizations   │
                    │  Alpha sweep comparisons     │
                    └──────────┬──────────────────┘
                               │ REST API
                    ┌──────────▼──────────────────┐
                    │     FastAPI Backend          │
                    │  /generate  /analyze  /info  │
                    └──────────┬──────────────────┘
                               │ PyTorch hooks
              ┌────────────────▼────────────────────┐
              │        Ministral 8B (bfloat16)       │
              │   Layer 18 ← steering hook injects   │
              │   SAE decoder direction × strength   │
              └────────────────┬────────────────────┘
                               │
              ┌────────────────▼────────────────────┐
              │      Custom BatchTopK SAE            │
              │  16,384 features, k=64               │
              │  Trained on StarCoderData             │
              │  Layers 18 + 27 of Ministral 8B       │
              └─────────────────────────────────────┘
```

**Steering mechanism:** During autoregressive generation, a forward hook on layer 18 adds `strength * W_dec[feature_idx]` to the residual stream at each decode step. The hook fires only during single-token decode, leaving prompt prefill unmodified. This is the standard SAE steering approach — the decoder vector points in the direction the model "moves" when a feature is active, and amplifying it makes the model move more in that direction.

---

## The Discovery Pipeline

Finding features that actually steer (not just detect) is the hard part. Our pipeline:

```
1. GENERATE        Ministral 8B produces code with and without target properties
   5 prompt pairs per property, 300 tokens each, temperature 0.3
                            ↓
2. CAPTURE          Forward pass through generated outputs, capture layer 18 + 27
   activations     residual stream via PyTorch hooks
                            ↓
3. ENCODE           SAE encodes each token position → 16,384-dim sparse activations
                    Per-token, NOT mean-pooled (preserves positional signal)
                            ↓
4. RANK             Differential score = target_freq × target_act − control_freq × control_act
                    Filter: diff > 0, cross-prompt frequency ≥ 0.4
                    Take top 5 per layer per property → 50 candidates
                            ↓
5. VERIFY           Steer generation on 3 neutral prompts at strengths 3, 5, 8
                    Pass criteria: property visible, output coherent, effect on ≥ 2/3 prompts
                    → 30 verified features shipped to the web UI
```

Runtime: ~36 minutes on one H100.

### Why generation-based discovery matters

Earlier approaches tried finding code features by running code *inputs* through the model and seeing which features fire. This finds detection features — features that recognize code patterns. But detection ≠ generation. A feature that fires when the model *reads* `List[int]` may have nothing to do with *writing* `List[int]`.

Our method captures activations during the model's own generation, comparing generations that exhibit the target property against matched generations that don't. Features found this way are part of the generation circuit by construction.

---

## The Full Experimental Journey

### Round 1: Community SAE on Mistral 7B (failed)

Started with `tylercosgrove/mistral-7b-sparse-autoencoder-layer16`. Profiled it: 80% dead features, TopK-128 constant sparsity, MLP-output hook point, trained on general text. Found 20 features via differential analysis on 1,278 code prompts across 8 languages. Every feature labeled by Mistral API as "code stub format" — they were detecting the HumanEval prompt structure, not code properties.

Tried steering with these features. Typing features at +500: gibberish. At +50: switched to JavaScript, still no types. At +10: slightly different code, no types. **Zero type annotations at any strength.**

### Round 2: Custom SAE on Mistral 7B (partial success)

Trained a BatchTopK SAE on StarCoderData targeting the residual stream. Found feature 304: 1,499x specificity for typed code, actually produced type annotations when steered. But 52% dead features limited the feature space.

### Round 3: Full pipeline on Ministral 8B (success)

Built a proper experiments pipeline. Trained custom SAEs at layers 18 and 27. Used generation-based contrastive discovery. Found 30 verified features across 5 properties with visible, coherent steering effects. The pipeline runs end-to-end in ~2 hours: generation → evaluation → activation capture → SAE training → feature discovery → steering verification.

### Round 4: Quantitative steering evaluation

Ran HumanEval pass-rate evaluation with contrastive and SAE steering at alpha = +/-3. Key finding: **all 22 steering conditions reduced pass rate below baseline.** Steering shifts code style without improving correctness. This is mechanistically interesting — the model's "code style" and "code correctness" representations are entangled at the feature level, and amplifying style features comes at a correctness cost.

---

## Web UI

The frontend provides three views:

- **Code tab**: Side-by-side diff of baseline vs steered generation. Alpha sweep scrubber to compare steering intensities from -2 to +3.
- **Analysis tab**: Token-level heatmap of feature activations. Click any token to see its SAE decomposition (top 10 active features) and layer-by-layer attribution (attention vs MLP contribution at each layer).
- **Dashboard tab**: Multi-feature activation timeline, activation matrix heatmap, and feature distribution histograms.

---

## Running It

### Prerequisites
- GPU VM with 2x H100 (or equivalent with 48+ GB VRAM)
- Python 3.10+, Node.js 18+
- SAE checkpoints on the VM (`~/8b_saes/layer_18_sae_checkpoint.pt`)

### Backend (on GPU VM)
```bash
cd backend
SAE_CHECKPOINT=~/8b_saes/layer_18_sae_checkpoint.pt \
  uvicorn backend.server:app --host 0.0.0.0 --port 8000
```

### Frontend (local)
```bash
npm install
npm run dev   # → localhost:3000
```

### SSH tunnel (connects local frontend to VM backend)
```bash
ssh -L 8000:localhost:8000 -i ~/.ssh/id_rsa azureuser@20.38.0.252
```

### Running the discovery pipeline
```bash
# On the GPU VM
python scripts/run_discovery.py   # ~36 min, produces results/demo_features.json
```

---

## Project Structure

```
.
├── backend/                FastAPI server (steering + analysis endpoints)
│   └── server.py           Ministral 8B + custom TopK SAE, PyTorch hooks
├── src/                    Next.js frontend
│   ├── app/page.tsx        Main UI (feature sliders, diff view, analysis tabs)
│   └── components/         Visualization components (heatmaps, charts, timelines)
├── experiments/            ML pipeline (generation → evaluation → SAE → steering)
│   ├── sae/                Custom SAE architecture (TopK, BatchTopK)
│   ├── steering/           Steering hook implementation
│   ├── generation/         vLLM runner, activation capture
│   ├── evaluation/         Sandboxed code execution, pass/fail judging
│   └── scripts/            Pipeline orchestration (00-05)
├── scripts/                Feature discovery (run_discovery.py, demo_diff.py)
├── results/                Experimental outputs
│   ├── demo_features.json  30 verified features with scores and strengths
│   ├── ranking.json        Full Phase 2 ranking data
│   ├── mistral-7b/         Mistral 7B generation + steering results
│   └── ministral-8b/       Ministral 8B generation + steering results
├── FEATURES_FOUND.md       Discovered features (30 verified, 5 properties)
├── STEERING_ANALYSIS.md    Why the community SAE failed (root cause analysis)
└── SAE_TRAINING_PLAN.md    Custom SAE training methodology
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Model | Ministral 8B Instruct (bfloat16, 2x H100 NVL) |
| SAE | Custom BatchTopK (d_sae=16,384, k=64, trained on StarCoderData) |
| SAE training | sae_lens + custom TopK implementation |
| Steering | PyTorch `register_forward_hook()` on residual stream |
| Backend | FastAPI (generate, analyze, feature registry endpoints) |
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| Evaluation | HumanEval + MBPP, sandboxed execution, 6 prompt variants |
| Infrastructure | Azure VM, 2x NVIDIA H100 NVL (96 GB VRAM each) |
