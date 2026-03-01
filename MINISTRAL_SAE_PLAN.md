# SAE Training Plan: Ministral-8B-Instruct-2410

## Goal

Train a BatchTopK SAE on `mistralai/Ministral-8B-Instruct-2410` for code feature steering, replacing the current Mistral 7B + custom SAE stack.

---

## Why Ministral-8B?

Ministral-8B (October 2024) is a newer, more capable model than Mistral-7B-Instruct-v0.1 (September 2023):
- 36 layers (vs 32) — more capacity for feature specialization
- 128K context (vs ~8K effective) — better long-code understanding
- Interleaved sliding-window + full attention — efficient long-range dependencies
- Trained on more recent data with improved instruction tuning
- Same hidden dimension (d_model=4096) — SAE architecture transfers directly

---

## Key Architectural Differences from Mistral 7B

| Parameter | Mistral 7B v0.1 | Ministral 8B 2410 |
|---|---|---|
| Parameters | 7.24B | 8.02B |
| Layers | 32 | **36** |
| d_model | 4,096 | 4,096 (same) |
| MLP intermediate | 14,336 | **12,288** (smaller) |
| Vocab size | 32,000 | **131,072** (4x larger) |
| Attention | Uniform sliding (4K window) | **Interleaved** (1 full + 3 sliding, 32K window) |
| Attention bias | No | **Yes** (Q/K/V/O projections have bias) |
| RoPE theta | 10,000 | **1e8** |
| Context | ~8K effective | **128K** |
| Tokenizer | SentencePiece v1 | **V3-Tekken** |
| License | Apache 2.0 (open) | **MRL-0.1** (research-only, gated) |

### Implications for SAE Training
- **d_model=4096 unchanged** → SAE dimensions (d_in=4096, d_sae=32768) carry over directly
- **36 layers** → layer 18 is the 50% depth point (analogous to layer 16 in 32-layer model)
- **Interleaved attention** → full-attention layers at indices 0, 4, 8, 12, **16**, **20**, 24, 28, 32 — prefer these for SAE placement
- **~16GB in bfloat16** → fits on one H100 with room for SAE training overhead

---

## Critical Constraint: No TransformerLens Support

**Ministral-8B is NOT supported by TransformerLens.** The interleaved sliding-window attention pattern and attention biases are not implemented.

This has two consequences:

### 1. SAE Training: Use `AutoModelForCausalLM` path in sae_lens

sae_lens supports training on any HuggingFace model via `model_class_name="AutoModelForCausalLM"`. Hook names must use HuggingFace parameter names instead of TransformerLens names:

| Concept | TransformerLens (Mistral 7B) | HuggingFace (Ministral 8B) |
|---|---|---|
| Residual stream post-layer N | `blocks.N.hook_resid_post` | `model.layers.N` |
| MLP output | `blocks.N.hook_mlp_out` | `model.layers.N.mlp` |
| Attention output | `blocks.N.hook_attn_out` | `model.layers.N.self_attn` |

### 2. Inference Steering: Rewrite Backend with PyTorch Hooks

The current backend uses `HookedTransformer` from TransformerLens for steering. We must replace this with native PyTorch `register_forward_hook()` on the HuggingFace model. This is a significant but straightforward change.

---

## Model Access

Ministral-8B uses the **Mistral Research License (MRL-0.1)**:
- Non-commercial / research use only
- Gated on HuggingFace — must accept license terms before download
- Requires `huggingface-cli login` with an authorized token

### Setup Steps
1. Go to https://huggingface.co/mistralai/Ministral-8B-Instruct-2410
2. Accept the gated access form (name, affiliation, acknowledgments)
3. On the GPU VM: `huggingface-cli login` with the authorized token
4. Verify: `python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('mistralai/Ministral-8B-Instruct-2410', device_map='cpu', torch_dtype='float16')"`

---

## Layer Selection

With 36 layers and interleaved attention (full at 0, 4, 8, 12, 16, 20, 24, 28, 32):

| Layer | Depth % | Attention Type | Rationale |
|---|---|---|---|
| 16 | 44% | **Full attention** | Near-middle, analogous to Mistral-7B layer 16; captures semantic features |
| 18 | 50% | Sliding | Exact midpoint; sliding attention may have different feature profile |
| 20 | 56% | **Full attention** | Slightly past middle; closer to output, may have more generation-causal features |

**Recommendation: Train on layer 16 first** (full-attention, closest analog to the working Mistral-7B setup). If results are weak, try layer 20 as fallback.

---

## Training Configuration

### SAE Architecture

```python
from sae_lens.saes import BatchTopKTrainingSAEConfig

sae_cfg = BatchTopKTrainingSAEConfig(
    d_in=4096,                    # Ministral-8B hidden dimension (same as Mistral 7B)
    d_sae=32768,                  # 8x expansion
    k=64,                         # Target L0 ~64
    dtype="float32",              # SAE weights in float32 for stability
    aux_loss_coefficient=1/32,    # AuxK dead neuron revival
    decoder_init_norm=0.1,
)
```

### Runner Configuration

```python
from sae_lens import LanguageModelSAERunnerConfig
from sae_lens.config import LoggingConfig

logger_cfg = LoggingConfig(
    log_to_wandb=True,
    wandb_project="ministral-8b-code-sae",
    wandb_log_frequency=100,
    eval_every_n_wandb_logs=10,
)

runner_cfg = LanguageModelSAERunnerConfig(
    sae=sae_cfg,
    logger=logger_cfg,

    # Model — AutoModelForCausalLM path (NOT TransformerLens)
    model_name="mistralai/Ministral-8B-Instruct-2410",
    model_class_name="AutoModelForCausalLM",
    model_from_pretrained_kwargs={"torch_dtype": "bfloat16"},

    # Hook point — HuggingFace naming for residual stream post-layer 16
    hook_name="model.layers.16",

    # Dataset — starcoderdata, streamed
    dataset_path="bigcode/starcoderdata",
    streaming=True,
    is_dataset_tokenized=False,
    context_size=512,
    prepend_bos=True,

    # Training
    training_tokens=2_000_000_000,   # 2B tokens
    train_batch_size_tokens=4096,
    lr=5e-5,
    lr_scheduler_name="constant",
    lr_warm_up_steps=1000,
    dead_feature_window=5000,
    dead_feature_threshold=1e-8,

    # Eval
    n_eval_batches=10,

    # Checkpoints
    n_checkpoints=5,
    checkpoint_path="checkpoints/ministral_sae_v1",
    save_final_checkpoint=True,
    output_path="checkpoints/ministral_sae_v1",

    # Hardware
    dtype="bfloat16",                # Model activations in bfloat16 (native dtype)
    device="cuda:1",
    seed=42,
)
```

### Key Differences from Mistral 7B Training Config

| Parameter | Mistral 7B | Ministral 8B | Why |
|---|---|---|---|
| `model_name` | `mistralai/Mistral-7B-Instruct-v0.1` | `mistralai/Ministral-8B-Instruct-2410` | New model |
| `model_class_name` | (default/TransformerLens) | `"AutoModelForCausalLM"` | No TransformerLens support |
| `hook_name` | `blocks.16.hook_resid_post` | `model.layers.16` | HuggingFace naming |
| `dtype` | `float16` | `bfloat16` | Ministral native dtype; bfloat16 avoids overflow |
| `model_from_pretrained_kwargs` | `{"dtype": "float16"}` | `{"torch_dtype": "bfloat16"}` | HF kwarg name differs |
| `wandb_project` | `mistral-7b-code-sae` | `ministral-8b-code-sae` | Separate tracking |
| `checkpoint_path` | `checkpoints/code_sae_v1` | `checkpoints/ministral_sae_v1` | Separate checkpoints |

### VRAM Budget

| Component | Estimated VRAM |
|---|---|
| Ministral-8B in bfloat16 | ~16 GB |
| SAE training (32K × 4096 × float32 × 2 matrices) | ~1 GB |
| Activations + optimizer states | ~8-12 GB |
| **Total** | **~25-29 GB** |
| Available (H100 GPU 1) | 96 GB |

Plenty of headroom. Could increase batch size or dictionary size if desired.

### Training Time Estimates

| Tokens | Est. Time (H100) | Quality |
|---|---|---|
| 50M | ~5-10 min | Pipeline test only |
| 500M | ~1-2 hours | Minimum viable |
| 2B | ~4-6 hours | Good quality (target) |
| 4B | ~8-12 hours | Production quality |

Note: Ministral-8B is ~10% larger than Mistral-7B, so expect slightly slower throughput per token.

---

## Training Script

Create `scripts/train_ministral_sae.py` — adapted from the existing `scripts/train_sae.py`:

### Changes from `train_sae.py`
1. `model_name` → `"mistralai/Ministral-8B-Instruct-2410"`
2. Add `model_class_name="AutoModelForCausalLM"` to runner config
3. `hook_name` → `"model.layers.16"` (HuggingFace naming)
4. `dtype` → `"bfloat16"` (native dtype for Ministral)
5. `d_in` stays 4096 (hidden dim is the same)
6. Update print statements and wandb project name
7. Update checkpoint default path to `checkpoints/ministral_sae_v1`
8. Verify HuggingFace gated access works before training starts

### Dataset Note

The `bigcode/starcoderdata` dataset uses `"content"` as its text column. With the `AutoModelForCausalLM` path, sae_lens may handle column naming differently. The existing script renames `"content"` → `"text"` via `override_dataset`. This approach should be preserved.

However, the new Tekken tokenizer (131K vocab) will tokenize code differently than Mistral 7B's SentencePiece tokenizer. This is expected and means the SAE will learn features aligned with Ministral's tokenization, which is what we want.

---

## Post-Training: Feature Discovery

### Challenge

The existing feature discovery scripts (`scripts/explore_trained_sae.py`, `scripts/typing/`, `scripts/prompt_observe/`) all use TransformerLens (`HookedTransformer`) for model inference and activation collection. These **will not work** with Ministral-8B.

### Solution: Rewrite with PyTorch Hooks

Replace `HookedTransformer.run_with_cache()` with native PyTorch forward hooks on the HuggingFace model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Ministral-8B-Instruct-2410",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Ministral-8B-Instruct-2410")
sae = SAE.load_from_disk("checkpoints/ministral_sae_v1/final", device="cuda:0")

# Collect activations via PyTorch hook
activations = {}
def capture_hook(module, input, output):
    # output is (hidden_states, ...) for MistralDecoderLayer
    if isinstance(output, tuple):
        activations["residual"] = output[0].detach()
    else:
        activations["residual"] = output.detach()

target_layer = model.model.layers[16]
handle = target_layer.register_forward_hook(capture_hook)

# Forward pass
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    model(**inputs)

# Encode with SAE
sae_acts = sae.encode(activations["residual"])
handle.remove()
```

### Adapted Feature Discovery Pipeline

The contrastive analysis approach remains the same — only the activation collection mechanism changes:
1. **Collect activations**: Use PyTorch hooks instead of `model.run_with_cache()`
2. **Encode with SAE**: `sae.encode()` is framework-agnostic, works the same
3. **Contrastive scoring**: Pure numpy/torch, no framework dependency
4. **Feature labeling**: Mistral API calls, no framework dependency
5. **Steering verification**: Use PyTorch hooks for steering (see next section)

---

## Backend Changes

### Current Backend (TransformerLens)
```python
# server.py — current approach
model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, ...)
with model.hooks(fwd_hooks=[(hook_point, steering_hook)]):
    steered_text = model.generate(...)
```

### New Backend (PyTorch Hooks)
```python
# server.py — new approach for Ministral-8B
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Ministral-8B-Instruct-2410",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Ministral-8B-Instruct-2410")
sae = SAE.load_from_disk("checkpoints/ministral_sae_v1/final", device="cuda:0")

# Steering hook — modifies residual stream in-place
def make_steering_hook(sae, feature_overrides):
    steering_vectors = []
    for feat_id, strength in feature_overrides:
        vec = sae.W_dec[feat_id].detach().clone()
        steering_vectors.append((strength, vec))

    def hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        for strength, vec in steering_vectors:
            hidden_states[:, :, :] += strength * vec.to(hidden_states.device, hidden_states.dtype)
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    return hook

# Usage
target_layer = model.model.layers[16]
handle = target_layer.register_forward_hook(make_steering_hook(sae, overrides))
output = model.generate(input_ids, max_new_tokens=300, temperature=0.3, do_sample=True)
handle.remove()
```

### Key Differences
- `model.hooks()` context manager → `register_forward_hook()` + `handle.remove()`
- `model.generate(prompt_string)` → `model.generate(input_ids)` (need explicit tokenization)
- Hook receives `(module, input, output)` instead of `(value, hook)`
- Output may be a tuple `(hidden_states, attention, ...)` — must handle correctly

---

## Execution Plan

### Phase 0: Prerequisites (30 min)

```bash
# On local machine / HuggingFace website
# 1. Accept Ministral-8B gated access at:
#    https://huggingface.co/mistralai/Ministral-8B-Instruct-2410

# On GPU VM
ssh -i ~/.ssh/id_rsa azureuser@20.38.0.252
conda activate steering

# 2. Verify HuggingFace auth for gated model
huggingface-cli login  # if not already
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    'mistralai/Ministral-8B-Instruct-2410',
    torch_dtype='bfloat16',
    device_map='cpu'
)
print(f'Loaded: {sum(p.numel() for p in model.parameters()):,} parameters')
print(f'Layers: {len(model.model.layers)}')
print(f'd_model: {model.config.hidden_size}')
"

# 3. Verify sae_lens AutoModelForCausalLM support
python -c "
from sae_lens import LanguageModelSAERunnerConfig
cfg = LanguageModelSAERunnerConfig.__init__.__doc__
print('model_class_name' in str(LanguageModelSAERunnerConfig.__dataclass_fields__))
"

# 4. Wandb login
wandb login

# 5. Verify GPU 1 free
nvidia-smi
```

### Phase 1: Test Run (15 min)

```bash
screen -S ministral_sae

# Copy training script to VM
# scp scripts/train_ministral_sae.py azureuser@20.38.0.252:~/

# Short test: 50M tokens
python train_ministral_sae.py --tokens 50_000_000 --checkpoint-path checkpoints/ministral_sae_test

# Verify:
# - Model loads correctly on GPU 1
# - Hook point "model.layers.16" is found
# - Training loop runs without errors
# - Loss decreases
# - Checkpoint is saved
```

### Phase 2: Full Training (4-6 hours)

```bash
# 2B tokens — good quality initial run
python train_ministral_sae.py \
    --tokens 2_000_000_000 \
    --checkpoint-path checkpoints/ministral_sae_v1

# Detach screen: Ctrl-A, D
# Monitor: screen -r ministral_sae
# GPU check: watch -n 5 nvidia-smi
```

### Phase 3: Evaluation (1-2 hours)

```bash
# Profile SAE quality
python explore_ministral_sae.py \
    --sae-path ~/checkpoints/ministral_sae_v1/final \
    --device cuda:0

# Expected metrics:
# - Dead features: <10% (target), <25% (acceptable)
# - L0 mean: ~64 with natural variance
# - Decoder norms: ~uniform
```

### Phase 4: Feature Discovery (1-2 hours)

Run adapted contrastive analysis pipeline on Ministral-8B + new SAE:
- Type annotations (typed vs untyped)
- Error handling (try/except vs simple)
- Functional style (map/lambda vs imperative)
- Recursion (recursive vs iterative)
- Verbose comments (documented vs minimal)

### Phase 5: Backend Integration (1-2 hours)

1. Rewrite `backend/server.py` to use `AutoModelForCausalLM` + PyTorch hooks
2. Update feature registry with newly discovered features
3. Test web UI end-to-end
4. Verify steering effects in the browser

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| **Gated access denied** | Accept license form on HuggingFace; verify token before training |
| **sae_lens hook name wrong** | Test with `python -c "print(dict(model.named_modules()).keys())"` to find exact names |
| **AutoModelForCausalLM path broken** | Fall back to `model_class_name="HookedTransformer"` if TL adds support; or use sae_lens cache-acts approach |
| **bfloat16 issues in sae_lens** | Fall back to `float16`; Ministral supports both |
| **OOM on GPU 1** | Reduce `train_batch_size_tokens` to 2048 or `context_size` to 256 |
| **Larger model = slower training** | Budget 50% more time than Mistral 7B estimates |
| **Dataset column mismatch** | Use `override_dataset` with renamed column as in existing script |
| **Features don't transfer** | Different model = different feature space; must redo full discovery pipeline |
| **PyTorch hook output format** | `MistralDecoderLayer.forward()` returns tuple; test actual output structure first |
| **Layer 16 too shallow/deep** | Try layer 20 (also full-attention) as alternative |

---

## Files to Create/Modify

| File | Action | Description |
|---|---|---|
| `scripts/train_ministral_sae.py` | **Create** | Training script adapted for Ministral-8B |
| `scripts/explore_ministral_sae.py` | **Create** | SAE profiling using PyTorch hooks instead of TransformerLens |
| `scripts/ministral_feature_discovery.py` | **Create** | Contrastive feature discovery using PyTorch hooks |
| `backend/server.py` | **Modify** | Replace TransformerLens with HuggingFace model + PyTorch hooks |
| `MINISTRAL_SAE_PLAN.md` | **Create** | This plan document |

---

## Timeline

| Phase | Duration | What |
|---|---|---|
| Prerequisites | 30 min | Accept license, verify access, setup |
| Test run | 15 min | 50M tokens to verify pipeline |
| Full training | 4-6 hours | 2B tokens on GPU 1 |
| Evaluation | 1-2 hours | Profile SAE quality metrics |
| Feature discovery | 1-2 hours | Contrastive analysis + labeling |
| Backend rewrite | 1-2 hours | PyTorch hooks, test steering |
| Integration test | 30 min | Web UI end-to-end |
| **Total** | **~8-14 hours** | Setup through working web UI |

If the 2B-token run produces good results, ship it. If marginal, extend to 4B tokens (+4-6 hours).
