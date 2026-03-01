# Pipeline Infrastructure Plan

## Decisions and Reasoning

### Activation capture: generated tokens only, layer 16 only

**Choice**: Capture activations only at generated token positions, not prompt tokens. Capture only layer 16, not layer 24.

**Alternatives considered for token scope**:
- *Full sequence (prompt + generation)*: Would let us study how prompt representation affects features, but roughly doubles storage (~100GB → ~200GB per layer) and mixes prompt-variant signal into SAE training data. The SAE would learn "prompt format" features rather than "code structure" features.
- *Last token only*: Tiny storage (~2GB total) but loses all token-level resolution. Only useful for generation-level classification, not mechanistic interpretability. We couldn't say "this feature fires on assert statements."

**Why generated tokens only**: We want token-level features ("this feature fires on type annotations") without paying for prompt tokens that just encode the task description. ~50GB is manageable. The generated tokens are where the model's "decisions" about code structure live.

**Why layer 16 only, not also layer 24**: Layer 16 is at the 50% point (16/32) where models build the richest semantic representations — this is where structural properties like typedness and decomposition should be most clearly encoded. Layer 24 (75%) is more output-oriented, with features mapping to token predictions rather than structural concepts. Capturing a second layer doubles storage (~50GB → ~100GB) and doubles the downstream work (contrastive vectors per layer, SAE per layer, analysis per layer) for uncertain additional value. If layer 16 results are weak, we can re-run capture for layer 24 — the generation and evaluation data is already saved.

### Inference strategy: vLLM generate → HuggingFace teacher-forcing

**Choice**: Two-phase approach. Phase 1: vLLM generates all outputs fast. Phase 2: HuggingFace forward pass with hooks captures activations.

**Key insight**: Activations at position *i* depend only on tokens 0..*i* (causal attention mask). Whether you generate autoregressively or feed the full sequence at once, **the activations at each generated token position are identical** given the same token sequence. This means teacher-forcing captures the exact same activations as autoregressive generation, but as a single forward pass (~50ms) instead of 600 sequential steps.

**Why not generate + capture in one pass**: vLLM is optimized for throughput (PagedAttention, continuous batching) but doesn't expose residual stream hooks. HuggingFace exposes hooks via `register_forward_hook` but is slower for generation. Using each tool for what it's best at gives us the best of both.

**Why not TransformerLens**: We don't need its capabilities. TL re-implements the model from scratch to make *every* intermediate tensor hookable — attention patterns, QKV decomposition, per-head outputs, MLP internals, named hook points at ~6 positions per layer across all 32 layers. That's the toolkit for circuit-level analysis (tracing information flow through specific attention heads and MLPs). We need the block output (residual stream) at 1 layer — for activation capture, HF provides this via `output_hidden_states=True` with zero hooks needed. For steering, we need one `register_forward_hook` to modify the residual stream during generation. TL would add a dependency with version constraints, a model re-initialization step (it translates weights into its own `HookedTransformer` architecture), and an abstraction layer — all to give us capabilities we won't use.

TL's speed disadvantage is a secondary concern, and the explanation is more nuanced than "no flash attention." TL prioritizes observability of all intermediates — to expose attention patterns, it cannot use fused attention kernels (like flash attention) that never materialize the N×N attention matrix. But this only affects attention pattern hooks, *not* residual stream hooks. You CAN have flash attention and residual stream access simultaneously (which is exactly what HF gives us). TL's slower speed comes from the broader design choice of keeping everything hookable, which constrains it to less-fused kernels throughout the model, plus its generation path being less optimized than HF's (which has battle-tested KV caching). For teacher-forcing forward passes, TL is only ~2-3× slower — feasible on 2 GPUs. The gap compounds during autoregressive steering (hundreds of sequential steps), but even there, the primary reason to avoid TL is the same: we don't need what it provides.

**Why not SAELens**: Same reasoning. SAELens provides battle-tested training loops, dead-feature handling, logging/visualization, and compatibility with TL's activation format. We need to train one SAE on one set of activations. That's an encoder (Linear → TopK), decoder (Linear), and MSE + aux loss — ~100 lines of PyTorch. SAELens would add a dependency chain (SAELens → TransformerLens → specific torch version) to provide capabilities we won't use. **Contingency**: If we end up iterating on SAE variants or need analysis tooling beyond our custom scripts, we can reintroduce SAELens later.

### Variants: 6 conditions (dropped naming discipline + data-structure choice)

**Choice**: Baseline, typed, invariants, decomposition, error_handling, control_flow.

**What we dropped and why**:
- *Naming discipline* (e.g., "use descriptive variable names"): Most cosmetic of the variants. Variable names in the output are a surface property — the model's internal computation doesn't fundamentally change because you name a variable `window_sum` vs `ws`. Also hardest to measure structurally (what counts as "descriptive"?). Would only be valuable with a paired ablation (force good names vs force bad names), which we decided against for simplicity.
- *Canonical data-structure choice* (e.g., "prefer dict" vs "forbid dict"): Architecturally different from other variants — all others add structural *requirements* ("add type hints"), but this constrains *content* ("use dict"). It's also task-specific (only tasks involving counting/frequency benefit from dict), reducing effective sample size. The paired ablation design (force vs forbid) would be the only variant with that structure, complicating analysis.

**Why single variants, not combinations**: Combinations (e.g., typed+invariants) are scientifically interesting for interaction effects, but multiply conditions. Adding even 3 combinations pushes from 20k to 30k generations. We can study interactions post-hoc by looking at which SAE features fire under multiple variants. The marginal information from explicit combinations doesn't justify the compute cost on a 12-hour timeline.

### Output contract: universal constraint on all variants

**Choice**: "Return only code. Define exactly the function with the specified signature. No prints. No markdown." is applied to every condition including baseline.

**Why**: If the contract were one of the compared variants, we'd be measuring format compliance AND structural effects simultaneously — a confound. By making it universal, the baseline is "contract only" and each variant adds exactly one structural property. This means the measured delta between baseline and variant is the pure marginal effect of that structural property.

### Sampling: temp=0.7, top_p=0.95, same for all 3 runs

**Choice**: Identical sampling config across all 3 runs per task×variant.

**Alternatives considered**:
- *Different temperatures per run* (e.g., greedy + 2 stochastic): Would make pass rate estimates per variant uninterpretable — "is variant X better, or did it benefit from the greedy run?"
- *temp=0.0 greedy, 3 identical runs*: Deterministic. All 3 runs produce identical output, making the 3 runs pointless. No variance for SAE training diversity.

**Why temp=0.7**: Standard for pass@k evaluation. Gives meaningful variance across runs (some pass, some fail on the same task). This variance is exactly what the SAE needs — the contrast between successful and failing activation traces is what lets it isolate feature directions correlated with correctness.

### Data splits: all tasks for SAE, HumanEval for steering

**Choice**: All 1,138 tasks (HumanEval + MBPP) used for generation + SAE training. HumanEval (164 tasks) used exclusively as the steering test set.

**Why HumanEval for steering**: The SAE trains on *activations*, not on tasks. Activations from HumanEval and MBPP live in the same model activation space (both are code generation through Mistral 7B). Features learned on MBPP activations fire on HumanEval activations just fine. HumanEval's lower pass rate (38% vs MBPP's 50%) is actually *better* for steering — more room to show improvement, tasks sit right at the capability boundary where representation changes have the most leverage. And HumanEval is the canonical eval, making results directly comparable to published work.

### SAE (stretch goal): TopK, 32k features, K=64, stratified training on layer 16

**Architecture choice**: TopK over ReLU because TopK enforces exact sparsity (no L1 penalty tuning), gives better reconstruction quality, and is the current standard (Anthropic's published SAE work). K=64 means 64 features active per token position.

**Why 32k features, not 65k**: With a 2M token training budget, each feature sees ~2M × (64/32k) = ~4k activation events. Sufficient for 32k features. 65k would give ~2k events per feature — borderline, with more dead features. 32k means faster analysis, fewer dead features, and we only need 3 steering candidates.

**Stratified training**: Sample activation tokens 50/50 from pass and fail generations. Without this, the SAE would mostly see activations from easier MBPP tasks (50% pass rate) — learning "common code patterns" rather than "patterns that distinguish success from failure." Implementation: maintain separate token pools for pass/fail, interleave during batch construction.

**Token budget**: 2M tokens (out of ~6M total). Sufficient for 32k features, keeps training under 30 minutes on a single H100.

**Why layer 16**: It's at the 50% point (16/32) where models build the richest semantic representations. Layer 24 (75%) is more output-oriented — features there tend to map to token predictions rather than structural concepts. Since we're looking for features corresponding to *structural properties* (types, invariants, decomposition), we want the representational layer.

### Steering approach: SAE primary, contrastive vectors as fallback

**Choice**: Train an SAE on captured activations and use SAE feature directions for steering (primary path). Compute contrastive vectors (difference-in-means between variant and baseline activations) as a guaranteed fallback if SAE training is slow, features are hard to interpret, or results are unclear.

**Why SAE primary**: The SAE decomposes the activation space into sparse, specific features that may be more surgically effective for steering than coarse contrastive directions. It enables discovery of features we didn't design variants for (surprise findings). It also produces the stronger scientific story — "we found specific internal features that correspond to structural properties" is more mechanistically informative than "we found that the average typed-vs-baseline direction works for steering."

**Why keep contrastive as fallback**: Contrastive vectors are trivial to compute from the same captured activations — no training step, no hyperparameters, no dead-feature risk. They're interpretable by construction ("this is the typed direction" = mean(typed activations) − mean(baseline activations) at layer 16). If SAE analysis hits any snag, contrastive vectors guarantee a working demo with zero additional compute. They can also serve as a comparison: do SAE features steer more precisely than the coarse contrastive direction?

### Steer layer: layer 16

**Choice**: Inject steering vectors (both contrastive and SAE-based) at layer 16.

**Why layer 16**: It's where we capture activations and train the SAE. Contrastive vectors are computed in this space, so no projection is needed. Standard practice is to steer at the same layer the directions were derived from.

### Steering: decode-only (not prefill)

**Choice**: During `generate()`, only inject the steering vector when `hidden_states.shape[1] == 1` (decode steps), not during the initial prompt prefill.

**Why not steer during prefill**: With KV caching, the prompt is processed in one forward pass (prefill), then each new token is generated one at a time (decode). If we inject during prefill, we modify the prompt's hidden states, which alters the KV cache entries for all prompt tokens. This permanently distorts how the model attends to the prompt — potentially misunderstanding the task, misinterpreting the signature, or attending to wrong parts of the requirements. We'd be testing "does this direction change how the prompt is interpreted?" rather than "does this direction change how code is generated?" — a confound.

**Why Anthropic steers during prefill for their experiments**: Their features are often semantic world-concepts (Golden Gate Bridge, deception). Steering during prefill intentionally shifts how the model interprets *everything* — that's the goal. Our features are generation structure properties (typedness, invariants, decomposition). We want to nudge generation decisions while keeping task comprehension intact.

**Implementation**: The condition is trivial. During decode, HF's KV-cached `generate()` passes one token at a time through each layer, so `hidden_states.shape[1] == 1`. During prefill, `hidden_states.shape[1] == prompt_length`. The hook checks this:
```python
def steering_hook(module, input, output):
    hidden_states = output[0]
    if hidden_states.shape[1] == 1:  # decode step only
        hidden_states = hidden_states + alpha * direction
    return (hidden_states,) + output[1:]
```

**Note**: With KV caching, there is no "repeated perturbation of old tokens." Each token's hidden state is computed exactly once — during prefill for prompt tokens, during its decode step for generated tokens. The concern about repeated perturbation only applies without KV cache (where the model would recompute the full sequence at each step).

### Alpha values: +3.0, -3.0

**Choice**: One positive (amplify) plus one negative (sign-flip control), plus a random direction at matched norm as specificity control. Three conditions per direction.

**Why simplified from {1.0, 3.0, 5.0, -3.0}**: With contrastive vectors as the primary path, we have 5 variant directions (typed, invariants, decomposition, error_handling, control_flow). At 3 conditions each × 164 tasks = 2,460 generations. Adding more alpha values would push into 4-5h territory. The sign-flip at -3.0 is the essential causal control — if +3.0 improves pass rate and -3.0 hurts it, that's strong causal evidence. The random control confirms specificity. We can sweep additional alpha values as a stretch goal.

**Plus random-direction control**: For each variant direction, also test a random direction at matched norm (α=3.0). If random directions at the same injection magnitude don't help, the effect is specific to the discovered structural direction.

### Token ID preservation

**Choice**: Store exact token IDs from vLLM generation, including chat template special tokens. Never re-tokenize text for teacher-forcing replay.

**Why this is critical**: If we decode vLLM's output to text and re-tokenize for the teacher-forcing activation capture pass, differences in whitespace handling, special tokens, or BPE merges between vLLM's tokenizer and HF's tokenizer could silently change the token sequence. Teacher-forcing with wrong token IDs produces wrong activations — the captured activations wouldn't match what the model "saw" during generation. vLLM returns token IDs directly via `outputs[i].token_ids`; we store and re-use them verbatim.

### Timeout: 3 seconds

**Choice**: 3s timeout per test execution, not 10s.

**Why lowered**: Most correct code runs in <100ms. With ~20k executions, generous timeouts on hanging code create significant wall-clock risk (a few hundred 10s timeouts = 30+ minutes wasted). 3s is generous for any correct solution and limits worst-case to ~10 minutes of timeout overhead. Steering deltas are relative comparisons, so consistent timeout policy is what matters, not absolute fairness to slow solutions.

### Sharding: by task_id

**Choice**: When splitting work across 2 GPUs, shard by task_id (all variants and runs for a given task go to the same GPU), not by flat index across the task×variant×run matrix.

**Why**: Grouping by task_id lets each process reuse prompt tokenization and dataset metadata across the 6 variants × 3 runs for the same task. Minor efficiency gain, but more importantly it reduces the bug surface — each process handles complete task units rather than arbitrary slices.

### Code extraction: regex + compliance flag, greedy retry

**Choice**: Use regex to extract code from model output. Record `extraction_clean` flag. On failure, retry once with same prompt at temp=0.0 (greedy). If still fails, mark as `extraction_fail`.

**Why greedy retry**: Same prompt preserves the experimental condition. Greedy decoding produces the model's most likely output, which is almost always clean code — markdown wrapping is a stochastic artifact at temp=0.7. If greedy also fails, these are genuine cases where the model can't follow format instructions for this task×variant. Data loss is <5%.

**Why not a modified "code only please" retry prompt**: Would change the generation context, making activations from retries come from a different "condition". Contaminates the experimental design.

### Failure taxonomy: 7 categories

**Choice**: pass, wrong_answer, syntax_error, type_error, runtime_error, timeout, extraction_fail.

**Why granular over simple**: Distinguishing syntax errors from type errors from wrong answers lets us analyze variant-specific failure reduction. For example: does the "typed" variant specifically reduce type_errors? Does "error_handling" reduce runtime_errors on edge cases? A 4-category taxonomy (pass/wrong/crash/timeout) would obscure these signals. Implementation cost is minimal — just catching specific exception types.

### Sandbox: subprocess + 3s timeout

**Choice**: Run each test in a subprocess with 3s timeout and resource limits (ulimit).

**Why 3s, not 10s**: Most correct code runs in <100ms. With ~20k executions, generous timeouts on hanging code create significant wall-clock risk (a few hundred 10s timeouts = 30+ minutes wasted). 3s is generous for any correct solution and limits worst-case overhead to ~10 minutes. Steering deltas are relative comparisons, so consistent timeout policy matters more than absolute fairness to slow solutions.

**Why not Docker**: ~200ms overhead per execution. At 20k executions that's ~70 min of pure overhead. Subprocess is ~1ms overhead. For a hackathon on a controlled VM, subprocess + timeout is sufficient.

### Orchestration: 2 independent processes, pre-sharded

**Choice**: Split tasks into 2 chunks. Launch 2 independent Python processes, one per GPU (CUDA_VISIBLE_DEVICES). Merge results after.

**Alternatives considered**:
- *torch.distributed*: Adds process group setup, distributed saving logic. No benefit when tasks are independent.
- *Ray*: Better load balancing for heterogeneous workloads. But our tasks are homogeneous (same model, same max_tokens). Overkill for 2 GPUs.

**Why this works**: Mistral 7B fits on a single H100 (14GB fp16, 80GB available). Tasks are independent. If one process crashes, the other keeps running. Merge is trivial (concatenate JSON files). Dead simple, good enough.

### UI integration: defer

**Choice**: FastAPI server on VM, build later. May fake live steering for demo.

**Why defer**: The pipeline and results are the core deliverable. The UI is a presentation layer. Building live steering (user adjusts slider → model generates on VM) requires WebSocket infrastructure and careful state management. Pre-computing results and serving static JSON gives 90% of the demo value at 10% of the implementation cost.

## Generation Counts

**Main experiment**: 6 conditions × 3 runs × 1,138 tasks = **20,484 generations**
**SAE steering** (primary): 3 features × 3 conditions (+α, −α, random) × 164 tasks + 164 baseline = **1,640 generations**
**Contrastive steering** (fallback): 5 directions × 3 conditions × 164 tasks + 164 baseline = **2,624 generations**
**Total**: ~20,484 + 1,640 + retries (~5%) ≈ **~23,000 generations** (+ ~2,600 contrastive if needed as fallback)

## The 6 Experimental Conditions

All conditions share the universal output contract:
> "Return only code. Define exactly the function with the specified signature. No prints. No markdown."

| # | Condition | Added instructions |
|---|---|---|
| 1 | **baseline** | (contract only, no structural additions) |
| 2 | **typed** | "Use full Python type hints. The function must pass static type checking." |
| 3 | **invariants** | "Clearly state assumptions using assertions. Handle edge cases explicitly with assert or if+raise." |
| 4 | **decomposition** | "Use at least 2 helper functions OR ≥3 named intermediate variables. Avoid long one-liner expressions." |
| 5 | **error_handling** | "On invalid input, raise ValueError. Handle empty inputs explicitly. Never silently return wrong results." |
| 6 | **control_flow** | "Use guard clauses and early returns. Limit nesting depth to 2. Prefer flat control flow." |

## Pipeline Stages

### Stage 1: Generation (vLLM) — ~1h on 2×H100

Generate all 20,484 code outputs using 2 independent vLLM instances (1 per GPU, CUDA_VISIBLE_DEVICES).
Shard by task_id: each GPU handles ~569 tasks × 6 variants × 3 runs.
Mistral 7B fits on a single H100 (14GB fp16, 80GB available).
Save: prompt, generated code, **exact token IDs from vLLM** (never re-tokenize), seed.

**Token ID preservation**: vLLM returns token IDs directly via `outputs[i].token_ids`. Store these verbatim.
The teacher-forcing replay in Stage 3 must use these exact IDs. Re-tokenizing decoded text risks silent
sequence mismatches due to whitespace handling, special tokens, or BPE merge differences.

**Seed strategy**: `seed = base_seed + run_id` (42, 43, 44 for runs 0, 1, 2). Same seed across all tasks
within a run. Different prompts produce different outputs regardless — the seed only controls sampling
randomness given fixed logits. This is simpler to reproduce, debug, and ensures any systematic pattern
is attributable to the task/variant, not the seed.

**Retry workflow**: After batch generation, run extraction on all outputs (fast CPU regex). Collect
extraction failures (~5%, ~1000 records). Batch re-generate just the failures with vLLM at temp=0.0.
Run extraction on retries. Mark remaining failures as `extraction_fail`. Update records with retry's
token IDs and generated text (Stage 3 must use the final output, not the original failed one).
This keeps vLLM's batching advantage — the retry batch is small and takes only a few minutes.

### Stage 2: Test Execution — ~20-40 min (CPU-bound, can overlap with Stage 3)

Run generated code against HumanEval/MBPP test cases. Sandboxed subprocess, **3s timeout**.
Record: pass/fail, failure category, error message hash.

**HumanEval test harness**: Each task has an `entry_point` and a `check(candidate)` function. We define the
generated function and pass it to `check()`. The executor wraps this into a single executable script.

**MBPP test harness**: Each task has a `test_list` — raw assertion strings like `assert func_name(args) == expected`.
The function name in the assertions comes from the reference solution. We extract this function name from
`test_list` and use it in the prompt ("Write a Python function `{func_name}`..."). The executor wraps the
generated code + test assertions into one executable script. Some tasks have `test_setup_code` (imports) —
include those when present.

**Unified executor interface**: Both datasets produce a single "test script" string via a thin adapter.
The executor doesn't need to know which dataset — it just runs the script in a subprocess.

### Stage 3: Activation Capture (HF teacher-forcing) — ~30 min on 2×H100

Load Mistral 7B via HuggingFace `AutoModelForCausalLM`.
For each generation, feed prompt+generated_tokens (using stored token IDs) as a single forward pass
with `output_hidden_states=True`. Read `outputs.hidden_states[17]` (layer 16 block output).
Extract activations at generated-token positions only (slice using prompt length).
Batch size 16-32 per GPU. Save as memory-mapped files, sharded by GPU.

**Precise tensor definition**: "Residual stream at layer 16" = the block output of `MistralDecoderLayer` 16,
i.e., the hidden state after both residual additions (post-attention + post-MLP) within that block.
In Mistral's architecture, each block does:
```
x = x + self_attn(input_layernorm(x))     # post-attention residual
x = x + mlp(post_attention_layernorm(x))   # post-MLP residual = block output
```
This is the standard tensor used in SAE literature and is what `output_hidden_states` returns.

**Indexing**: `hidden_states[0]` = embedding output (before any decoder layer), `hidden_states[i+1]` = output
of decoder layer `i`. So layer 16 → index 17. **This off-by-one must be validated in the sanity check (see below).**

**Why `output_hidden_states` over hooks for capture**: More robust across model versions — no need to handle
tuple unpacking (`MistralDecoderLayer.forward()` returns `(hidden_states, attn_weights, present_key_value)`),
no risk of hooking the wrong module. Memory cost of returning all 33 hidden states: 33 × batch × seq × 4096 × 2 bytes
= ~2.6GB per batch at batch=16, seq=600. Trivial on 80GB H100. We index layer 16 and discard the rest.
**Fallback**: If memory spikes due to intermediate buffers, switch to a single `register_forward_hook` on
`model.model.layers[16]` — captures only one layer's tensor, minimal memory.

### Stage 4: SAE Training — ~30 min on 1×H100

Train custom TopK SAE (32k features, K=64) on layer 16 activations.
Implementation: ~100 lines PyTorch (encoder, TopK activation, decoder, MSE + aux loss).
No SAELens dependency.

**Stratified training**: Sample activation tokens equally from pass and fail generations (50/50 balance).
Without this, the SAE would mostly see activations from easier MBPP tasks (50% pass rate),
learning "common code patterns" rather than "patterns that distinguish success from failure."
Implementation: maintain separate token pools for pass/fail, alternate batches or interleave.

**Token budget**: ~2M tokens for training. At ~300 generated tokens per generation and ~20k generations,
the full pool is ~6M tokens. 2M is sufficient for 32k features (each feature sees ~2M × 64/32k ≈ 4k events)
and keeps training fast.

### Stage 5: SAE Feature Analysis — ~10 min

For each of 32k features, compute:
- Mean activation on pass vs fail (effect size = Cohen's d)
- Mean activation per variant (variant correlation)
- Top activating token snippets (interpretability via gen_token_ids)
Select top 3 features by success effect size, preferring features that also correlate with a structural variant.

### Stage 6: SAE Steering Experiment — ~30-45 min on 2×H100

For each of 3 SAE candidate features, on HumanEval 164 tasks:
- α = +3.0 (amplify, using SAE decoder direction d_f)
- α = -3.0 (sign-flip control)
- Random SAE feature at matched norm (specificity control)

3 features × 3 conditions × 164 tasks = **1,476 generations**.
Plus 164 baseline generations (no steering) = **1,640 total**.

Uses HuggingFace `generate()` with decode-only steering hook at layer 16.
Batch size 8-16. 1,640 × 600 tokens / 2 GPUs / ~200 tok/s ≈ 30-45 min.
Record pass/fail + code + steering metadata.

**Steering hook (decode-only)**: Attach to `model.model.layers[16]`.
Only inject during decode steps (new token generation), not during prompt prefill:
```python
def steering_hook(module, input, output):
    hidden_states = output[0]
    if hidden_states.shape[1] == 1:  # decode step only, not prefill
        hidden_states = hidden_states + alpha * direction
    return (hidden_states,) + output[1:]
```
This avoids distorting the model's comprehension of the task prompt.
See "Steering: decode-only" decision above for full rationale.

### Stage 7 (fallback): Contrastive Vectors + Steering

**Use if SAE analysis is slow, features are hard to interpret, or results are unclear.**

Compute difference-in-means steering directions for each variant vs baseline:
```
d_typed = mean(activations[variant=="typed"]) - mean(activations[variant=="baseline"])
d_invariants = mean(activations[variant=="invariants"]) - mean(activations[variant=="baseline"])
... (5 directions total)
```
Each direction is a single vector of shape `(4096,)`. No training, no hyperparameters, interpretable by construction.

Then run the same steering protocol: 5 directions × 3 conditions × 164 tasks + 164 baseline = **2,624 generations** (~1-1.5h).
Uses the same decode-only hook as Stage 6.

**Also useful as comparison**: If SAE steering works, contrastive steering shows whether SAE features
steer more precisely than the coarse variant-level direction.

### Stage 8 (later): UI Integration

FastAPI server on VM exposing results as static JSON. Next.js dashboard for feature browsing + steering visualization. May fake live steering for demo.

## File Structure

```
experiments/
├── README.md                    # Environment setup, script execution order, output locations
├── config.py                    # All parameters: variants, alpha values, paths, seeds, timeouts
├── datasets/
│   ├── __init__.py
│   ├── load_humaneval.py        # Load HumanEval, normalize to common Task format
│   └── load_mbpp.py             # Load MBPP, normalize to common Task format
├── prompts/
│   ├── __init__.py
│   ├── variants.py              # 6 variant prompt templates
│   └── builder.py               # Combines task + variant + universal contract into final prompt
├── generation/
│   ├── __init__.py
│   ├── vllm_runner.py           # Batch generation with vLLM, saves outputs + exact token IDs
│   └── activation_capture.py    # HF teacher-forcing forward pass, output_hidden_states on layer 16
├── evaluation/
│   ├── __init__.py
│   ├── extractor.py             # Regex code extraction + compliance detection + greedy retry
│   ├── executor.py              # Sandboxed subprocess execution with 3s timeout
│   └── judge.py                 # Failure classification: 7-category taxonomy
├── storage/
│   ├── __init__.py
│   ├── schema.py                # GenerationRecord dataclass, serialization
│   └── activation_store.py      # Memory-mapped activation storage, sharded by GPU
├── sae/
│   ├── __init__.py
│   ├── model.py                 # TopK SAE implementation (~100 LOC PyTorch)
│   ├── train.py                 # Stratified training loop (balanced pass/fail sampling)
│   ├── analyze.py               # Feature-success correlation, variant correlation
│   └── select_candidates.py     # Top-k feature selection for steering
├── contrastive/                 # (fallback)
│   ├── __init__.py
│   ├── compute_directions.py    # Difference-in-means vectors: variant vs baseline
│   └── analyze.py               # Variant pass rates, structural measurements
├── steering/
│   ├── __init__.py
│   ├── hook.py                  # Decode-only steering hook (shape[1]==1 gate)
│   ├── run_sae.py               # SAE feature steering experiment (primary)
│   ├── run_contrastive.py       # Contrastive vector steering experiment (fallback)
│   └── analyze_steering.py      # Compute deltas, controls, significance tests
├── scripts/
│   ├── 00_sanity_check.py       # Run all 5 sanity checks before pipeline
│   ├── 01_generate.py           # Shard by task_id across 2 GPUs, run vLLM generation
│   ├── 02_evaluate.py           # Run test execution on all generations
│   ├── 03_capture_activations.py # HF teacher-forcing activation capture, layer 16 only
│   ├── 04_train_sae.py          # Train SAE on collected activations
│   ├── 05_sae_steering.py       # SAE-based steering experiments
│   ├── 06_contrastive.py        # (fallback) Contrastive directions + steering
│   ├── 07_analyze_all.py        # Final analysis + export results for UI
├── tests/
│   ├── test_extractor.py        # Code extraction regex, compliance detection, retry logic
│   ├── test_executor.py         # Sandbox execution, timeout enforcement, signal handling
│   ├── test_prompt_builder.py   # Prompt construction, variant templates, universal contract
│   ├── test_activation_store.py # Mmap write/read round-trip, sharding, dtype preservation
│   ├── test_sae_model.py        # TopK forward/backward, sparsity, reconstruction loss
│   ├── test_steering_hook.py    # Tuple handling, decode-only gating, α=0 no-op
│   └── conftest.py              # Shared fixtures: tiny model, sample records, temp dirs
└── requirements.txt             # vllm, transformers, torch, datasets, etc.
```

## Per-Generation Record Schema

```python
@dataclass
class GenerationRecord:
    # Identity
    task_id: str              # "humaneval_023" or "mbpp_412"
    dataset: str              # "humaneval" or "mbpp"
    variant_id: str           # "baseline", "typed", "invariants", etc.
    run_id: int               # 0, 1, 2
    seed: int                 # Random seed for exact replay

    # Prompt
    prompt_text: str          # Full assembled prompt
    prompt_tokens: int        # Token count

    # Generation
    generated_text: str       # Raw model output
    extracted_code: str       # Regex-extracted code (empty if extraction failed)
    gen_token_ids: list[int]  # Token IDs of generated sequence
    generated_tokens: int     # Token count

    # Extraction
    extraction_clean: bool    # Whether first extraction succeeded
    extraction_retried: bool  # Whether a greedy retry was attempted
    retry_succeeded: bool     # Whether retry extraction succeeded

    # Evaluation
    passed: bool              # All tests passed
    failure_category: str     # pass|wrong_answer|syntax_error|type_error|runtime_error|timeout|extraction_fail
    error_message: str        # Raw error message (if any)
    error_hash: str           # Hash of error message for clustering

    # Activation metadata (activations stored separately in mmap files)
    activation_file: str      # Path to mmap shard file
    activation_offset: int    # Byte offset within shard
    activation_length: int    # Number of token positions
```

## Activation Storage Format

Each GPU writes its own shard file (layer 16 only):
```
activations/
├── layer16/
│   ├── shard_0.mmap          # GPU 0's activations, shape (N_tokens, 4096), float16
│   ├── shard_1.mmap          # GPU 1's activations
│   └── ...
└── index.json                # Maps (task_id, variant_id, run_id) → (shard, offset, length)
```

Memory-mapped files allow the SAE training loop to stream through activations without loading everything into RAM.
Total storage: ~6M tokens × 4096 × 2 bytes ≈ ~50GB.

### Storage Strategy: NVMe scratch → persistent OS disk

The NC80adis VM has 2× 3.5TB local NVMe drives, RAID0'd and mounted at `/scratch` (7TB total, free with the VM).
These are ephemeral — data is lost on VM deallocation (not reboot).

**During pipeline execution**: All intermediate and output data writes go to `/scratch`:
- `/scratch/generations/` — JSONL shards (`shard_0.jsonl`, `shard_1.jsonl`), one line per GenerationRecord, appendable and crash-safe
- `/scratch/activations/` — memory-mapped numpy files (`shard_0.npy`, `shard_1.npy`), float16, shape (N_tokens, 4096). Each GenerationRecord stores `activation_file`, `activation_offset`, `activation_length` to index in.
- `/scratch/sae/` — SAE checkpoints (`model.pt`), training log (`training_log.jsonl`)
- `/scratch/steering/` — steering results as JSONL (`sae_steering.jsonl`, `contrastive_steering.jsonl`), same GenerationRecord format with added steering metadata
- `/scratch/analysis/` — final outputs (`pass_rates.csv`, `feature_candidates.json`)

**Why JSONL for records**: ~20k records total — no need for SQLite query overhead or Parquet's write-once constraint. JSONL is appendable (one complete record per line), human-readable for debugging, and trivially shardable across GPU processes. Load into DataFrame for analysis.

**After pipeline completes**: Copy final results to the persistent OS disk:
```bash
rsync -av /scratch/generations/ ~/results/generations/
rsync -av /scratch/sae/ ~/results/sae/
rsync -av /scratch/steering/ ~/results/steering/
# Skip raw activations (~50GB) unless needed — they're regenerable from stored token IDs
```

**Why**: NVMe is significantly faster for the large sequential writes during activation capture
and the random reads during SAE training. The OS disk (256GB) is too small for raw activations
anyway. Final results (generations JSON, SAE weights, steering results) are small (~1-2GB)
and fit easily on the OS disk for persistence.

## Prompt Construction

### HumanEval format (preserves canonical format):
```
[INST]
{docstring_with_signature}

Requirements:
- Return only the function implementation.
- No prints, no markdown, no explanation.
{variant_instructions}
[/INST]
def {function_name}({args}):
```

### MBPP format (preserves natural language):
```
[INST]
{task_description}

Write a Python function `{function_name}` that solves this.

Requirements:
- Return only the function implementation.
- No prints, no markdown, no explanation.
{variant_instructions}
[/INST]
```

Note: The `[INST]...[/INST]` wrapper is Mistral's chat template. Exact template applied via tokenizer.apply_chat_template().

## Steering Experiment Design

### Primary: SAE feature steering

For each of top 3 SAE features by success effect size (d_f = SAE decoder direction):
```
Conditions per task (HumanEval 164 tasks):
  1. baseline:    no intervention (shared across all features, run once)
  2. amplify:     h' = h + 3.0 · d_f    at layer 16 (decode only)
  3. suppress:    h' = h + (-3.0) · d_f  at layer 16 (decode only)
  4. random_ctrl: h' = h + 3.0 · d_rand  at layer 16 (matched norm, decode only)
```

164 baseline + 3 features × 3 conditions × 164 tasks = **1,640 steering generations**.

### Fallback: Contrastive vector steering

For each contrastive direction d (typed, invariants, decomposition, error_handling, control_flow):
```
Same conditions as above, using d = mean(variant activations) - mean(baseline activations).
```

5 directions × 3 conditions × 164 tasks + 164 baseline = **2,624 generations**.

### Shared details

All steering uses HuggingFace `generate()` with a **decode-only** `register_forward_hook` on
`model.model.layers[16]`. The hook only fires when `hidden_states.shape[1] == 1` (single-token decode step),
not during prompt prefill. This keeps task comprehension intact and isolates the effect to generation decisions.
Batch size 8-16 on 2×H100.

## Claim Structure

| Evidence tier | Source | What you show | Claim you can make |
|---|---|---|---|
| Variant pass rates | Stages 1-2 | Typed/invariant/decomposition variants change pass rate | "Structural prompt properties measurably affect code generation success" |
| SAE feature correlation | Stages 4-5 | SAE features correlate with both success and specific variants | "Internal feature directions correspond to interpretable structural properties" |
| SAE steering | Stage 6 | Amplifying a feature helps, suppressing hurts | "Feature direction causally influences success, not just correlational" |
| Random control | Stage 6 | Random features at matched norm don't help | "Effect is specific to the discovered feature, not generic activation injection" |
| Structural measurement | Stage 6 | Steered code has more type hints / asserts / helpers | "Steering produces the intended structural change, not just 'any improvement'" |
| Contrastive comparison (fallback/stretch) | Stage 7 | Contrastive directions also work for steering | "Variant-level representation differences are causally active, and SAE decomposes them into finer features" |

## Sanity Checks (run as `scripts/00_sanity_check.py` before any full pipeline stage)

These cost minutes and prevent training an SAE on the wrong tensor or steering the wrong layer.

**Check 1: Hidden state indexing**
Run a tiny batch (2 prompts, 32 tokens) with `output_hidden_states=True`. Also attach a
`register_forward_hook` to `model.model.layers[16]`. Compare `outputs.hidden_states[17]` with
the hooked tensor. They must be identical (not just close — exactly equal, since no computation
diverges). If they differ, the indexing is wrong. This catches the off-by-one and verifies
that "block output" means the same thing in both code paths.

**Check 2: Shape validation**
Confirm `hidden_states[17].shape == (batch, seq_len, 4096)`. If shape is wrong, the model
architecture doesn't match our assumptions. Also verify `hidden_states` has 33 entries
(1 embedding + 32 layers) for Mistral 7B.

**Check 3: Teacher-forcing determinism**
Run the same token sequence through the model twice. Activations must be bitwise identical.
If not, there's a nondeterminism issue (dropout not disabled, or stochastic kernel behavior).
Set `model.eval()` and `torch.manual_seed()` before testing.

**Check 4: Steering hook — tuple handling + decode-only gating**
Attach the decode-only steering hook to `model.model.layers[16]`. Confirm the hook receives a
tuple and `output[0]` has the expected shape. Run `generate()` with the hook active on a short
prompt and verify:
- At α=0 (no-op): output is identical to unhooked generation (proves the hook doesn't corrupt)
- At large α: output visibly differs (proves the hook is modifying activations)
- The hook only fires on decode steps: add a counter inside the hook; after generating N tokens
  on a prompt of length P, the hook should have fired P+N times total but only injected on N
  of those (the decode steps where `shape[1]==1`)

**Check 5: Token ID round-trip**
Take a vLLM generation output, feed its exact token IDs through HF tokenizer.decode(), then
re-tokenize with HF tokenizer.encode(). Confirm the token IDs match. If they don't, there's a
tokenizer mismatch between vLLM and HF. This must be resolved before running activation capture.

## Tests

Tests run before committing to full pipeline runs. They catch bugs in the components that are hardest to debug once the pipeline is in flight (extraction regexes, sandbox edge cases, activation storage integrity).

### Unit Tests

All tests use `pytest`. No GPU required — tests mock or use tiny models where needed.

**`tests/test_extractor.py`** — Code extraction is the most fragile part of the pipeline. If extraction silently fails, we get phantom failures that look like model failures.
- Extracts code from well-formed markdown blocks (` ```python ... ``` `)
- Extracts when model omits language tag (` ``` ... ``` `)
- Extracts bare code (no markdown at all) — falls back to full text
- Detects compliance flag: output starts with function def matching expected signature
- Handles model outputs with explanation text before/after the code block
- Retry case: extraction fails → retry prompt → extraction succeeds on retry output
- Edge case: multiple code blocks → takes the first one (or the one containing the function name)
- Edge case: empty output → extraction_clean=False, extracted_code=""

**`tests/test_executor.py`** — Sandbox correctness is a safety requirement. A broken sandbox can hang the pipeline or produce misleading pass/fail results.
- Passing code: returns passed=True, failure_category="pass"
- Syntax error: returns passed=False, failure_category="syntax_error"
- Runtime error (NameError, TypeError): returns passed=False, failure_category="runtime_error"
- Timeout (infinite loop): returns passed=False, failure_category="timeout" within 3s + small margin
- Wrong answer (assertion fails): returns passed=False, failure_category="wrong_answer"
- Import not allowed (if sandboxed): returns appropriate failure category
- Verifies subprocess isolation: executor doesn't crash if generated code calls `sys.exit()`

**`tests/test_prompt_builder.py`** — Prompt construction is deterministic and should be easy to test, but bugs here silently corrupt all downstream data.
- Each variant produces a prompt containing the expected structural instruction
- Universal output contract appears in all 6 variant prompts
- Prompt includes the task description and test cases
- HumanEval and MBPP tasks produce different prompt formats (HumanEval has entry_point, MBPP has test_list)
- Seed doesn't affect prompt construction (prompts are deterministic, seeds only affect generation)

**`tests/test_activation_store.py`** — Activation storage bugs are catastrophic: if shapes or dtypes silently corrupt, SAE training produces garbage and we won't know until steering fails.
- Write a batch of float16 tensors, read them back, assert bitwise equality
- Mmap file can be opened read-only after writing
- Offset and length fields in GenerationRecord correctly index into the mmap file
- Multiple writes (simulating sharded capture) produce a valid concatenated file
- Shape is preserved: write (N, 4096), read back (N, 4096), not accidentally flattened

**`tests/test_sae_model.py`** — The SAE is custom code (~100 LOC), not a library. Bugs here waste an entire training run.
- Forward pass: input shape (batch, 4096) → output shape (batch, 4096)
- TopK sparsity: exactly K non-zero features in the latent representation
- Reconstruction: on random input, loss decreases after a few training steps (not stuck)
- Encoder/decoder weight shapes match: encoder is (4096, 32768), decoder is (32768, 4096)
- Gradient flows through both encoder and decoder

**`tests/test_steering_hook.py`** — The steering hook is the causal intervention mechanism. If it's wrong, the entire experiment is invalid.
- Hook receives a tuple from MistralDecoderLayer, returns a tuple
- At α=0: output[0] is identical to input (no-op)
- Decode-only gating: when shape[1] > 1 (prefill), no modification applied
- Decode-only gating: when shape[1] == 1 (decode), modification applied
- Direction vector is added with correct sign and magnitude
- Hook doesn't modify output[1:] (KV cache, attention weights)

### Smoke Test (`scripts/00_smoke_test.py`)

Runs on the actual VM with GPU before any pipeline stage. Takes ~2 minutes. Covers the 5 sanity checks defined above plus an end-to-end micro-run:

1. All 5 sanity checks (hidden state indexing, shape validation, teacher-forcing determinism, steering hook correctness, token ID round-trip)
2. **Micro end-to-end**: Take 2 HumanEval tasks, 1 variant (baseline), 1 run → generate with vLLM → extract code → execute in sandbox → capture activations → verify the full record is valid. This catches integration issues between components (e.g., token ID format mismatch between vLLM output and HF input).

Exits with code 0 on success, code 1 with a clear error message on failure. Must pass before running `01_generate.py`.

## Documentation Policy

Documentation lives in the code, with two standalone files:

- **`experiments/README.md`**: How to run the pipeline — environment setup, script execution order, where outputs land. Written before implementation. The entry point for anyone picking up the repo.
- **`experiments/RESULTS.md`**: Written after experiments run. Documents findings: selected features, steering deltas, pass rate tables. Separates results from the code that produced them.
- **Module-level docstrings**: Every `.py` file starts with a 1-3 line docstring explaining what it does and its role in the pipeline. Example: `"""Batch generation with vLLM. Produces GenerationRecords with exact token IDs for downstream teacher-forcing."""`
- **Type hints everywhere**: All function signatures fully typed. The types are the documentation — `def capture(records: list[GenerationRecord], model: PreTrainedModel) -> np.ndarray` says more than a paragraph.
- **`config.py` is self-documenting**: Every parameter has a comment explaining its value and why. This is the single source of truth for all magic numbers.
- **`argparse` help text on scripts**: Each `scripts/*.py` has `--help` that explains what the script does, what it reads, and what it writes. This is the user-facing documentation.
- **No wiki, no docs/ folder**: The plan.md is the design doc. The README is the run doc. The code is the implementation doc.

## Implementation Order

**Core path (must finish):**
1. `config.py` + `storage/schema.py` — Foundation, everything depends on these
2. `datasets/load_humaneval.py` + `datasets/load_mbpp.py` — Load and normalize tasks
3. `prompts/variants.py` + `prompts/builder.py` — Construct prompts for all conditions
4. `tests/test_prompt_builder.py` — Verify prompt construction before generation
5. `evaluation/extractor.py` + `evaluation/executor.py` + `evaluation/judge.py` — Test all outputs
6. `tests/test_extractor.py` + `tests/test_executor.py` — Verify extraction and sandbox before pipeline runs
7. `generation/vllm_runner.py` — Generate all outputs with exact token ID storage
8. `generation/activation_capture.py` + `storage/activation_store.py` — Capture layer 16 activations
9. `tests/test_activation_store.py` — Verify mmap storage round-trip before full capture
10. `scripts/00_sanity_check.py` — Validate indexing, shapes, determinism, hook gating, token IDs (includes smoke test micro end-to-end)
11. `scripts/01_generate.py` through `scripts/03_capture_activations.py` — Wire and run stages 1-3
12. `sae/model.py` + `sae/train.py` — Custom TopK SAE with stratified training
13. `tests/test_sae_model.py` — Verify TopK sparsity, gradient flow before training run
14. `sae/analyze.py` + `sae/select_candidates.py` — Feature analysis + candidate selection
15. `steering/hook.py` + `steering/run_sae.py` — Decode-only hook + SAE feature steering
16. `tests/test_steering_hook.py` — Verify tuple handling, decode-only gating, α=0 no-op before steering runs
17. `steering/analyze_steering.py` — Compute deltas, run tests, export results

**Fallback path (if SAE hits issues):**
18. `contrastive/compute_directions.py` — Compute difference-in-means directions (trivial, ~5 min)
19. `steering/run_contrastive.py` — Contrastive vector steering experiments
20. Same analysis pipeline as step 17

**Stretch (if time permits after core):**
21. Run contrastive steering as comparison to SAE steering
22. UI integration stub

## Dependencies (requirements.txt)

```
vllm>=0.4.0
transformers>=4.40.0
torch>=2.2.0
datasets>=2.19.0
numpy>=1.26.0
safetensors>=0.4.0
```

No TransformerLens or SAELens. We use HuggingFace `transformers` directly for all model operations
(activation capture via `output_hidden_states`, steering via `register_forward_hook`) and implement
a custom TopK SAE in ~100 lines of PyTorch for the stretch goal.

**Why no TL/SAELens**: For activation capture, HF provides `output_hidden_states=True` — zero hooks needed,
robust across model versions, correct by construction. For steering, we need one decode-only
`register_forward_hook` on a single decoder layer module. TL and SAELens are designed for circuit-level
interpretability (attention patterns, per-head decomposition, MLP analysis) which we don't do here.
See full reasoning in Decisions and Reasoning section above.

**Contingency**: If we later need per-head/MLP internals, standardized hook-point names, or SAE training
iteration tooling, TL/SAELens can be reintroduced. The activation data format (float16 tensors of shape
`(tokens, 4096)`) is framework-agnostic.

## Time Budget (12 hours, 2×H100)

### Core path (must finish)

| Phase | Wall clock | Notes |
|---|---|---|
| Build pipeline code | 3-4h | config, datasets, prompts, generation, evaluation, storage, SAE, steering |
| Sanity checks | 0.25h | scripts/00_sanity_check.py — validate before committing to full runs |
| Run generation (vLLM) | 1h | 20,484 generations on 2×H100, sharded by task_id |
| Run evaluation | 0.5h | 20,484 sandboxed test executions, 3s timeout (CPU, overlaps with capture) |
| Run activation capture | 0.5h | HF teacher-forcing, layer 16 only, on 2×H100 |
| Train SAE | 0.5h | Single GPU, 2M token budget, stratified pass/fail |
| SAE feature analysis | 0.25h | Correlation, candidate selection |
| Build steering code | 0.5h | Decode-only hook + SAE steering runner |
| Run SAE steering | 0.5-0.75h | 1,640 generations on 2×H100 |
| Analyze results + export | 0.5h | Pass rate deltas, structural measurements |
| **Core total** | **~8-10h** | Feasible with some margin |

### Fallback (if SAE hits issues)

| Phase | Wall clock | Notes |
|---|---|---|
| Compute contrastive vectors | 0.1h | Difference-in-means, trivial |
| Run contrastive steering | 1-1.5h | 2,624 generations on 2×H100 |
| **Fallback total** | **~1.5h** | Guaranteed demo if SAE path fails |

### Stretch (if core finishes early)

| Phase | Wall clock | Notes |
|---|---|---|
| Run contrastive steering (as comparison) | 1-1.5h | Compare SAE precision vs contrastive |
| UI integration stub | 1-2h | FastAPI + static results |
| **Stretch total** | **~2-3h** | |

### Scoping levers if time is tight
- Reduce to 2 SAE features instead of 3 — saves ~15min on steering
- Skip random controls, add them later — saves ~15min
- Fall back to contrastive immediately — saves SAE training+analysis time (~1h), guaranteed to work
