# Pipeline Plan

**Core pipeline**: Gather activations → Train SAE → Linear probe → SAE steering

**Current model**: Ministral-8B-Instruct-2410 (36 decoder layers, hidden dim 4096)
**Evaluation**: HumanEval (164 tasks) for steering, MBPP + HumanEval for SAE training
**Hardware**: 2× H100 NVL (96GB each)

## Decisions and Reasoning

### Activation capture: generated tokens only, multi-layer

**Choice**: Capture activations only at generated token positions, not prompt tokens. Capture at two layers — 50% and 75% of model depth. For Ministral-8B (36 layers): layers 18 and 27. Configurable per model via `config.py` `capture_layers` property.

**Alternatives considered for token scope**:
- *Full sequence (prompt + generation)*: Would let us study how prompt representation affects features, but roughly doubles storage (~100GB → ~200GB per layer) and mixes prompt-variant signal into SAE training data. The SAE would learn "prompt format" features rather than "code structure" features.
- *Last token only*: Tiny storage (~2GB total) but loses all token-level resolution. Only useful for generation-level classification, not mechanistic interpretability. We couldn't say "this feature fires on assert statements."

**Why generated tokens only**: We want token-level features ("this feature fires on type annotations") without paying for prompt tokens that just encode the task description. The generated tokens are where the model's "decisions" about code structure live.

**Why two layers**: Layer 18 (50%) captures mid-network semantic representations where structural properties should be most clearly encoded. Layer 27 (75%) captures late-network refinement where the model is committing to output decisions. Comparing both reveals whether steering is more effective at the representational or decisional stage. Our probe results confirm this matters: layer 27 achieves 0.870 AUC vs layer 18's 0.846.

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

### SAE: TopK, 16k features, K=64, stratified training, per-layer

**Architecture choice**: TopK over ReLU because TopK enforces exact sparsity (no L1 penalty tuning), gives better reconstruction quality, and is the current standard (Anthropic's published SAE work). K=64 means 64 features active per token position.

**Why 16k features (4× expansion), not 131k (32×)**: Our first attempt used 131k features and resulted in 83% dead features due to insufficient data-to-feature ratio (~11 tokens per feature). Reducing to 16k brought dead features down to ~10% while maintaining 84.5% variance explained. The 16k SAE has ~880 tokens per feature — well above the threshold for meaningful learning.

**Stratified training**: Sample activation tokens 50/50 from pass and fail generations. Without this, the SAE would mostly see activations from easier MBPP tasks (50% pass rate) — learning "common code patterns" rather than "patterns that distinguish success from failure." Implementation: maintain separate token pools for pass/fail, interleave during batch construction.

**Token budget**: 10M tokens (~7 epochs over 1.4M unique tokens). Keeps training under 30 minutes on a single H100.

**Per-layer training**: A separate SAE is trained for each capture layer (18 and 27). Each layer's activations live in a different region of activation space, so they need independent SAEs.

### Feature selection: Linear probe (not Cohen's d)

**Choice**: Train a logistic regression probe on SAE features to find which features jointly predict pass/fail. Project the probe weight vector back through the SAE decoder to get a steering direction in hidden-state space.

**Why probe over Cohen's d**: Cohen's d computes univariate effect sizes per feature. In practice, it selected near-dead features with inflated effect sizes (d=500k+) — features that fire a handful of times for one condition, giving technically huge but meaningless separation. Zero overlap between Cohen's d top features and probe top features. The probe finds features that are *jointly* predictive across the full dataset, handling redundancy and interactions.

**Why project through W_dec**: The probe learns a weight vector w ∈ R^{d_sae} in SAE feature space. The steering direction in hidden-state space is `direction = w @ W_dec` where W_dec is the SAE decoder matrix (d_sae, d_model). This gives a single (d_model,) vector that is the optimal linear combination of SAE decoder columns, weighted by predictive power.

**Direction sign**: The probe direction may point from pass→fail rather than fail→pass after projection through W_dec. We test both the original and flipped (negated) direction.

### Steering approach: SAE-projected probe direction primary, contrastive as comparison

**Choice**: The primary steering direction is the probe weight vector projected through the SAE decoder. Contrastive vectors (difference-in-means) serve as a comparison baseline. Individual SAE feature decoder rows are also tested but are expected to be weaker (too narrow/sparse).

**Why probe direction over individual features**: Individual SAE features are too narrow and sparse for effective steering. Our best single-feature delta was +1.2% (not significant). The probe combines all 16k features optimally, producing a direction norm of ~8-10 vs ~1.0 for individual features — a much stronger, more coherent signal.

### Steer layer: same as capture/SAE layer

**Choice**: Inject steering vectors at the same layer the SAE was trained on. For Ministral-8B: layer 18 or layer 27.

**Why match layers**: The SAE decoder columns define directions in the activation space of a specific layer. Steering at a different layer would require a projection that we don't have. Standard practice is to steer at the same layer the directions were derived from.

### Steering: decode-only (not prefill)

**Choice**: During `generate()`, only inject the steering vector when `hidden_states.shape[1] == 1` (decode steps), not during the initial prompt prefill.

**Why not steer during prefill**: With KV caching, the prompt is processed in one forward pass (prefill), then each new token is generated one at a time (decode). If we inject during prefill, we modify the prompt's hidden states, which alters the KV cache entries for all prompt tokens. This permanently distorts how the model attends to the prompt — potentially misunderstanding the task, misinterpreting the signature, or attending to wrong parts of the requirements. We'd be testing "does this direction change how the prompt is interpreted?" rather than "does this direction change how code is generated?" — a confound.

**Why Anthropic steers during prefill for their experiments**: Their features are often semantic world-concepts (Golden Gate Bridge, deception). Steering during prefill intentionally shifts how the model interprets *everything* — that's the goal. Our features are generation structure properties (typedness, invariants, decomposition). We want to nudge generation decisions while keeping task comprehension intact.

**Implementation**: The condition is trivial. During decode, HF's KV-cached `generate()` passes one token at a time through each layer, so `hidden_states.shape[1] == 1`. During prefill, `hidden_states.shape[1] == prompt_length`. The hook checks this:
```python
def steering_hook(module, input, output):
    hidden_states = output[0]
    if hidden_states.shape[1] == 1:  # decode step only
        hidden_states = hidden_states + alpha * direction.to(hidden_states.dtype)
    return (hidden_states,) + output[1:]
```

**Note**: The `.to(hidden_states.dtype)` cast is required — model runs in float16, direction is float32.

**Note**: With KV caching, there is no "repeated perturbation of old tokens." Each token's hidden state is computed exactly once — during prefill for prompt tokens, during its decode step for generated tokens. The concern about repeated perturbation only applies without KV cache (where the model would recompute the full sequence at each step).

### Alpha values: [0.5, 1.0, 1.5, 3.0, -1.5]

**Choice**: Wider sweep from gentle to aggressive, plus one negative as sign-flip control.

**Why this range**: With probe directions of norm ~8-10, aggressive alphas (3.0) at these norms overwhelm the residual stream (hidden states have norm ~3.8) and destroy performance. The signal lives at moderate alphas (0.5-1.5). The negative alpha (-1.5) is the essential causal control — if positive alpha improves and negative hurts, that's strong directional evidence.

**Lesson learned**: Our first experiments used only α=±3.0 and saw universal degradation. Expanding to include lower alphas revealed that the directions were valid but the injection magnitude was too high.

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

Load model via HuggingFace `AutoModelForCausalLM`.
For each generation, feed prompt+generated_tokens (using stored token IDs) as a single forward pass
with `output_hidden_states=True`. Read activations at capture layers (18 and 27 for Ministral-8B).
Extract activations at generated-token positions only (slice using prompt length).
Batch size 16-32 per GPU. Save as memory-mapped files, sharded by GPU.

**Precise tensor definition**: "Residual stream at layer N" = the block output of decoder layer N,
i.e., the hidden state after both residual additions (post-attention + post-MLP) within that block.

**Indexing**: `hidden_states[0]` = embedding output (before any decoder layer), `hidden_states[i+1]` = output
of decoder layer `i`. So layer 18 → index 19. **This off-by-one must be validated in the sanity check.**

**Multi-layer storage**: Each record stores an `activation_layers` dict keyed by layer number string,
with `{file, offset, length}` per layer. This replaces the old single-layer `activation_file`/`activation_offset`/`activation_length` fields.

### Stage 4: SAE Training — ~30 min on 1×H100 per layer

Train custom TopK SAE (16k features, K=64) per capture layer.
Implementation: ~100 lines PyTorch (encoder, TopK activation, decoder, MSE + aux loss).
No SAELens dependency.

**Stratified training**: Sample activation tokens equally from pass and fail generations (50/50 balance).
Without this, the SAE would mostly see activations from easier MBPP tasks (50% pass rate),
learning "common code patterns" rather than "patterns that distinguish success from failure."

**Token budget**: 10M tokens. ~7 epochs over 1.4M unique tokens. Sufficient for 16k features
(each feature sees ~10M × 64/16k ≈ 40k events).

### Stage 5a: SAE Feature Analysis — ~10 min per layer

For each feature, compute:
- Mean activation on pass vs fail (effect size = Cohen's d)
- Mean activation per variant (variant correlation)
- Top activating token snippets (interpretability via gen_token_ids)

**Note**: Cohen's d feature selection proved unreliable — it selected near-dead features with inflated
effect sizes (d=500k+). This stage is useful for understanding the SAE's feature space but not for
selecting steering candidates. The linear probe (Stage 5b) is used for feature selection instead.

### Stage 5b: Linear Probe — ~2 min on 1×GPU per layer

For each layer with a trained SAE:
1. Encode all generation records through the SAE (mean-pool across tokens → one d_sae vector per record)
2. Train logistic regression (PyTorch, GPU) on SAE features to predict pass/fail
   - BCE loss + L2 regularization (weight_decay=1e-3)
   - Adam optimizer, cosine LR schedule, 200 epochs, batch size 4096
3. Project probe weight vector through SAE decoder: `direction = w @ W_dec` → (d_model,) vector
4. Save as `probe_direction.pt` in contrastive format: `{"probe_pass_fail": tensor}`

This is the key step that bridges SAE decomposition → steering direction. The probe finds which features
are jointly predictive of the label, and the W_dec projection maps that back to the model's activation space.

### Stage 6: SAE Steering Experiment — ~30-45 min on 2×H100 per layer

For each probe direction, on HumanEval 164 tasks:
- Alphas: [0.5, 1.0, 1.5, 3.0, -1.5]
- Also test flipped (negated) direction if original doesn't improve pass rate
- Steer at the same layer the SAE was trained on

Uses HuggingFace `generate()` with decode-only steering hook.
2 GPU shards of 82 tasks each.

### Stage 7: Analysis

Compute pass rate deltas and Fisher's exact test for statistical significance.
Compare: baseline pass rate vs steered pass rate at each alpha.
p < 0.05 for significance.

### Contrastive Vectors (comparison, not fallback)

Compute difference-in-means steering directions for each variant vs baseline:
```
d_typed = mean(activations[variant=="typed"]) - mean(activations[variant=="baseline"])
... (5 directions total)
```
Each direction is a single vector of shape `(4096,)`. These serve as a comparison to the SAE-probe direction.
Tested alongside SAE directions in the same steering infrastructure.

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
│   ├── select_candidates.py     # Top-k feature selection for steering
│   └── probe.py                 # Linear probe on SAE features → steering direction
├── contrastive/                 # (fallback)
│   ├── __init__.py
│   ├── compute_directions.py    # Difference-in-means vectors: variant vs baseline
│   └── analyze.py               # Variant pass rates, structural measurements
├── steering/
│   ├── __init__.py
│   ├── hook.py                  # Decode-only steering hook (shape[1]==1 gate, dtype cast)
│   ├── run_experiment.py        # Unified steering experiment runner (SAE, probe, contrastive)
│   └── analyze_steering.py      # Compute deltas, Fisher's exact test, significance
├── scripts/
│   ├── run_stages_1_3.sh        # Generation, evaluation, activation capture
│   ├── run_stage_4_sae.sh       # SAE training per layer
│   ├── run_stages_4_7.sh        # Feature analysis + SAE/contrastive steering (stages 5-8)
│   └── run_probe_steer.sh       # Probe training + probe steering per layer
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
    activation_layers: dict   # {layer_num: {"file": str, "offset": int, "length": int}}
```

## Activation Storage Format

Each GPU writes its own shard file, per capture layer:
```
activations/
├── shard_0.npy               # GPU 0's activations, float16, multi-layer
├── shard_1.npy               # GPU 1's activations
└── ...
```
Each record's `activation_layers` dict maps layer number → `{file, offset, length}` to index into the shard files.

Memory-mapped files allow the SAE training loop to stream through activations without loading everything into RAM.
Total storage: ~6M tokens × 4096 × 2 bytes ≈ ~50GB.

### Storage Strategy: NVMe scratch → persistent OS disk

The NC80adis VM has 2× 3.5TB local NVMe drives, RAID0'd and mounted at `/scratch` (7TB total, free with the VM).
These are ephemeral — data is lost on VM deallocation (not reboot).

**During pipeline execution**: All intermediate and output data writes go to `/scratch/<model>/`:
- `/scratch/<model>/generations/` — JSONL shards, one line per GenerationRecord, appendable and crash-safe
- `/scratch/<model>/activations/` — memory-mapped numpy files, float16. Each record's `activation_layers` dict indexes into these.
- `/scratch/<model>/sae/layer_<N>/` — SAE checkpoints per capture layer
- `/scratch/<model>/steering/layer_<N>/` — steering results as JSONL, per layer
- `/scratch/<model>/analysis/layer_<N>/` — probe directions, feature stats, steering results per layer

**Why JSONL for records**: ~20k records total — no need for SQLite query overhead or Parquet's write-once constraint. JSONL is appendable (one complete record per line), human-readable for debugging, and trivially shardable across GPU processes. Load into DataFrame for analysis.

**After pipeline completes**: Copy final results to the persistent OS disk:
```bash
rsync -av /scratch/<model>/sae/ ~/results/<model>/sae/
rsync -av /scratch/<model>/steering/ ~/results/<model>/steering/
rsync -av /scratch/<model>/analysis/ ~/results/<model>/analysis/
# Skip raw activations (~50GB) — they're regenerable from stored token IDs
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

### Primary: Probe-projected SAE steering

For the probe direction projected through W_dec (per layer):
```
Conditions per task (HumanEval 164 tasks):
  1. baseline:         no intervention (run once)
  2. steer α=0.5:      h' = h + 0.5 · d_probe   at layer N (decode only)
  3. steer α=1.0:      h' = h + 1.0 · d_probe   at layer N (decode only)
  4. steer α=1.5:      h' = h + 1.5 · d_probe   at layer N (decode only)
  5. steer α=3.0:      h' = h + 3.0 · d_probe   at layer N (decode only)
  6. steer α=-1.5:     h' = h + (-1.5) · d_probe at layer N (decode only)
```

If results suggest the direction is inverted (negative alpha helps more than positive),
also test the flipped direction: d_probe_flipped = -d_probe.

164 baseline + 1 direction × 5 alphas × 164 tasks = **984 generations per layer**.
With 2 layers: ~1,968 total. With flipped direction: ~2,952 total.

### Comparison: Contrastive vector steering

Difference-in-means directions computed per variant. Tested alongside probe directions
using the same infrastructure.

### Shared details

All steering uses HuggingFace `generate()` with a **decode-only** `register_forward_hook` on
`model.model.layers[N]` (where N is the capture layer). The hook only fires when
`hidden_states.shape[1] == 1` (single-token decode step), not during prompt prefill.
2 GPU shards of 82 tasks each.

## Claim Structure

| Evidence tier | Source | What you show | Claim you can make |
|---|---|---|---|
| Variant pass rates | Stages 1-2 | Structural prompt properties change pass rate | "Structural instructions measurably affect code generation" |
| SAE probe signal | Stages 4-5b | Probe predicts pass/fail at 0.87 AUC from SAE features | "SAE features encode information predictive of code correctness" |
| SAE steering | Stage 6 | Probe direction causally changes pass rate | "The learned direction causally influences generation quality" |
| Sign control | Stage 6 | Positive alpha helps, negative hurts (or vice versa) | "Effect is directional, not generic perturbation" |
| Layer comparison | Stages 5b-6 | Different layers show different probe/steering quality | "Code quality is represented differently at different network depths" |
| Contrastive comparison | Stage 7 | Contrastive directions compared to probe | "SAE decomposition + probe finds more precise directions than raw difference-in-means" |

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

**Core pipeline:**
1. `config.py` + `storage/schema.py` — Foundation, everything depends on these
2. `datasets/load_humaneval.py` + `datasets/load_mbpp.py` — Load and normalize tasks
3. `prompts/variants.py` + `prompts/builder.py` — Construct prompts for all conditions
4. `evaluation/extractor.py` + `evaluation/executor.py` + `evaluation/judge.py` — Test all outputs
5. `generation/vllm_runner.py` — Generate all outputs with exact token ID storage
6. `generation/activation_capture.py` + `storage/activation_store.py` — Multi-layer activation capture
7. `sae/model.py` + `sae/train.py` — Custom TopK SAE with stratified training
8. `sae/analyze.py` — Feature-level analysis (Cohen's d, variant correlation)
9. `sae/probe.py` — **Linear probe on SAE features → steering direction** (the key new step)
10. `steering/hook.py` + `steering/run_experiment.py` — Decode-only hook + unified steering runner
11. `steering/analyze_steering.py` — Fisher's exact test, pass rate deltas

**Comparison directions (run alongside probe):**
12. `sae/select_candidates.py` — Individual feature selection (for comparison)
13. `contrastive/compute_directions.py` — Difference-in-means directions

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

## Current Results (Ministral-8B)

### SAE Training
- 16k features, K=64, 10M tokens, per-layer
- Layer 18: 84.5% variance explained, 10.6% dead features
- Layer 27: similar quality
- Previous 131k SAE attempt: 83% dead features (data-to-feature ratio too low)

### Linear Probe
- Layer 18: 77.0% accuracy, 0.846 AUC
- Layer 27: 79.6% accuracy, 0.870 AUC (better — later layer has more refined representations)
- Top features by probe weight have zero overlap with Cohen's d top features

### Probe Steering (original direction)
- Layer 18: All positive alphas degrade. α=3.0 gives -18.3% (p=0.0013). Direction points pass→fail.
- Layer 27: Mild degradation at positive alphas. α=-1.5 gives +1.2% (only positive delta). Suggests inversion.

### Probe Steering (flipped direction, HumanEval)
- Layer 27, negated direction, alphas [0.5, 1.0, 1.5, 3.0, -1.5]
- Best: α=1.5 → +1.2% (p=0.91, n=164). Right trend but not significant.

### Probe Steering (flipped direction, MBPP)
- Layer 27, negated direction, alphas [0.5, 1.0, 1.5]
- Baseline: 38.0% (370/974)
- Best: α=0.5 → +0.4% (p=0.89, n=974). Near-zero with 6× more data.

### Activation Patching (direction clamping)
- Clamp layer 27's projection onto probe direction to calibrated pass/fail targets
- Pass projection mean: ~-0.58, Fail projection mean: ~-0.96, Cohen's d ≈ 0.72
- Clamp to pass mean: -0.6% combined (n=164) — no improvement
- Clamp to fail mean: +0.0% combined — no degradation
- Zero direction: -3.0% — slight noise from removing information, not directional
- Only 2-6 tasks change outcome in any condition (out of 164)
- **Conclusion: probe direction is a readout, not a control signal.** Layer 27 encodes pass/fail information (0.87 AUC) but intervening on it does not change outcomes.

### Previous attempts
- 131k SAE: 83% dead features, universal degradation
- 16k SAE + Cohen's d: Best delta +1.2% (not significant). Cohen's d selected near-dead features.
- Contrastive directions: Some reached significance but all negative.

## Time Budget (12 hours, 2×H100)

### Per-layer pipeline timing (actual, Ministral-8B on 2×H100)

| Phase | Wall clock | Notes |
|---|---|---|
| Stages 1-3 (gen + eval + capture) | ~3h | 20k generations, multi-layer capture |
| Stage 4: SAE training | ~30min/layer | 16k features, 10M tokens |
| Stage 5a: Feature analysis | ~10min/layer | Cohen's d, variant correlation |
| Stage 5b: Linear probe | ~2min/layer | PyTorch GPU logistic regression |
| Stage 6: Probe steering | ~30min/layer | 1 direction × 5 alphas × 164 tasks, 2 GPU shards |
| Stage 7: Analysis | ~1min | Fisher's exact test |
