# SAE Training Plan

## Motivation

The community-trained SAE (`tylercosgrove/mistral-7b-sparse-autoencoder-layer16`) is fundamentally inadequate for steering code generation properties. Evidence from `STEERING_ANALYSIS.md`:

| Problem | Evidence |
|---|---|
| 80% dead features | 104,821 of 131,072 features never fire on diverse prompts |
| TopK-128 forces constant sparsity | Every token fires exactly 128 features — no variation, features compete for slots |
| Wrong hook point | `blocks.16.hook_mlp_out` (MLP output) instead of `blocks.16.hook_resid_post` (residual stream) |
| Trained on general data | `monology/pile-uncopyrighted` — minimal code representation |
| Short context | `context_size=256` tokens — too short for realistic code |
| Typing features don't work | Zero type annotations produced at any strength from +5 to +500 |

Training a new SAE with modern architecture, code-heavy data, and the correct hook point should resolve these issues.

---

## Architecture: BatchTopK

**Why BatchTopK over TopK:**
- SAEBench (2025) ranks BatchTopK best on the sparsity-fidelity Pareto frontier
- Unlike TopK (which forces exactly K features per token), BatchTopK enforces K features **on average across the batch**, allowing natural variation per token
- Some tokens genuinely need more features (complex code), others need fewer (simple punctuation)
- AuxK dead-neuron loss keeps features alive during training, directly addressing the 80% dead feature problem

**Configuration:**

| Parameter | Value | Rationale |
|---|---|---|
| Architecture | BatchTopK | Best sparsity-fidelity tradeoff (SAEBench 2025) |
| d_in | 4096 | Mistral 7B hidden dimension |
| d_sae | 32,768 | 8x expansion (standard for 7B models, down from 32x/131K) |
| k | 64 | Target L0 ~64 (down from 128; sparser = more interpretable features) |
| Hook point | `blocks.16.hook_resid_post` | Residual stream captures full layer output, not just MLP |
| AuxK coefficient | 1/32 (0.03125) | Standard value for dead neuron revival |
| Decoder init norm | 0.1 | Standard BatchTopK initialization |

**Why 8x expansion instead of 32x:**
- 32x (131K features) with 80% dead means only ~26K alive — effectively 6.4x
- 8x (32K features) with good training should keep >90% alive (~29K) — more usable features
- Smaller dictionary trains faster, uses less VRAM, and is easier to search
- Can scale up to 16x in a second run if 8x proves insufficient

---

## Hook Point: Residual Stream vs MLP Output

The current SAE hooks into `blocks.16.hook_mlp_out`, which captures only the MLP sublayer's contribution. The residual stream (`blocks.16.hook_resid_post`) captures the **full layer output** including attention + MLP + skip connection. This is the standard hook point for steering because:

1. Steering vectors added to the residual stream propagate through all subsequent layers
2. The residual stream is the "main highway" of information flow
3. SAEBench and most published steering results use residual stream hooks
4. The README's own code examples use `hook_resid_post`

---

## Dataset Strategy

### Primary: `codeparrot/github-code` (ungated, code-focused)

The model needs to learn code-specific features. Training on general text (like Pile) dilutes code representation. The existing repo data (~165 snippet pairs, ~1,280 HumanEval stubs) totals maybe 500K tokens — three orders of magnitude short of the billions needed for SAE training.

**`codeparrot/github-code`** is the best choice:
- 1TB+ of GitHub code, no access restrictions (ungated — no HuggingFace login needed)
- Streams directly via HuggingFace `datasets` — no disk space required for the dataset itself
- Can filter by language: Python, TypeScript, JavaScript, Rust, Go, Java
- Immediately available, no agreements to sign

**Fallback:** `monology/pile-uncopyrighted` (also ungated, but general text — less code)

### Dataset configuration

```python
dataset_path = "codeparrot/github-code"
streaming = True  # Streams from HuggingFace, no download needed
context_size = 512  # 2x the current SAE's 256; captures realistic function bodies
```

### Note on code vs prose mix

Using a code-only dataset is fine for this project. The web UI's prompts are all code-related ("Implement a binary search function"), so the SAE doesn't need to understand prose. If prose handling becomes important later, we can retrain on a mix of `codeparrot/github-code` + `monology/pile-uncopyrighted`.

---

## Training Configuration

### Compute Budget

| Resource | Available | Required |
|---|---|---|
| GPU | 2x H100 NVL 96GB | 1x H100 (GPU 1, since GPU 0 runs the backend) |
| VRAM for model | ~14.7 GB (float16) | Available on GPU 1 |
| VRAM for SAE training | ~5-10 GB (32K features, float32 SAE) | Available on GPU 1 |
| Total VRAM needed | ~25 GB | Well within 96 GB |
| Disk | 216 GB free | Sufficient for streaming dataset + checkpoints |
| RAM | 619 GB free | More than sufficient |

### Token Budget

| Tokens | Estimated Time (1x H100) | Quality |
|---|---|---|
| 500M | ~1 hour | Minimum viable (for initial testing) |
| 2B | ~3-4 hours | Good quality, sufficient for evaluation |
| 4B | ~6-8 hours | Production quality |
| 8B | ~12-16 hours | High quality, diminishing returns beyond this |

**Plan: Train on 2B tokens first.** Evaluate, then extend to 4B if results are promising.

### Hyperparameters

```python
from sae_lens import LanguageModelSAERunnerConfig, BatchTopKTrainingSAEConfig

sae_config = BatchTopKTrainingSAEConfig(
    d_in=4096,
    d_sae=32768,
    k=64,
    dtype="float32",
    aux_loss_coefficient=1/32,
    decoder_init_norm=0.1,
)

runner_config = LanguageModelSAERunnerConfig(
    sae=sae_config,
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    hook_name="blocks.16.hook_resid_post",
    hook_layer=16,
    dataset_path="codeparrot/github-code",
    streaming=True,
    context_size=512,
    training_tokens=2_000_000_000,  # 2B tokens
    train_batch_size_tokens=4096,
    lr=5e-5,
    lr_warm_up_steps=1000,
    lr_decay_steps=0,  # No decay for initial run
    dead_feature_window=5000,
    dead_feature_threshold=1e-8,
    log_to_wandb=True,
    wandb_project="mistral-7b-code-sae",
    wandb_log_frequency=100,
    eval_every_n_wandb_logs=10,
    n_checkpoints=5,
    checkpoint_path="checkpoints/code_sae_v1",
    dtype="float16",  # Model dtype
    device="cuda:1",  # Use GPU 1 (GPU 0 has the backend)
)
```

### Key hyperparameter choices explained

- **`lr=5e-5`**: Conservative learning rate. The existing SAE likely used a higher rate, contributing to dead features. Can increase to `1e-4` if loss plateaus.
- **`lr_warm_up_steps=1000`**: Prevents early training instability from killing features.
- **`dead_feature_window=5000`**: Check for dead features every 5000 steps. AuxK loss will revive them.
- **`context_size=512`**: Captures full function bodies. The existing SAE used 256, which truncates most real code.
- **`train_batch_size_tokens=4096`**: Standard batch size. With 512 context, that's 8 sequences per batch.
- **`device="cuda:1"`**: Train on GPU 1 while the backend continues running on GPU 0.

---

## Training Script

Create `scripts/train_sae.py`:

```python
#!/usr/bin/env python3
"""Train a BatchTopK SAE on Mistral 7B for code steering.

Usage:
    python train_sae.py [--tokens 2_000_000_000] [--device cuda:1]

Requires:
    - HuggingFace login for gated datasets (huggingface-cli login)
    - wandb login for experiment tracking (wandb login)
"""

import argparse
import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2_000_000_000)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--dataset", type=str, default="codeparrot/github-code")
    parser.add_argument("--d-sae", type=int, default=32768)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/code_sae_v1")
    args = parser.parse_args()

    print(f"Training SAE: d_sae={args.d_sae}, k={args.k}, tokens={args.tokens:,}")
    print(f"Device: {args.device}, Dataset: {args.dataset}")

    runner_config = LanguageModelSAERunnerConfig(
        # SAE architecture
        architecture="batch_topk",
        d_in=4096,
        d_sae=args.d_sae,
        activation_fn_kwargs={"k": args.k},

        # Model
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        hook_name="blocks.16.hook_resid_post",
        hook_layer=16,

        # Data
        dataset_path=args.dataset,
        streaming=True,
        context_size=args.context_size,

        # Training
        training_tokens=args.tokens,
        train_batch_size_tokens=4096,
        lr=args.lr,
        lr_warm_up_steps=1000,
        dead_feature_window=5000,
        dead_feature_threshold=1e-8,

        # Logging
        log_to_wandb=True,
        wandb_project="mistral-7b-code-sae",
        wandb_log_frequency=100,
        eval_every_n_wandb_logs=10,

        # Checkpoints
        n_checkpoints=5,
        checkpoint_path=args.checkpoint_path,

        # Hardware
        dtype="float16",
        device=args.device,
    )

    runner = LanguageModelSAETrainingRunner(runner_config)
    sae = runner.run()

    # Save final SAE
    output_path = f"{args.checkpoint_path}/final"
    sae.save_model(output_path)
    print(f"Training complete. SAE saved to {output_path}")

if __name__ == "__main__":
    main()
```

**Note:** The exact sae_lens API for BatchTopK may use either the `BatchTopKTrainingSAEConfig` class or the `architecture="batch_topk"` parameter in `LanguageModelSAERunnerConfig`. The script above uses the simpler `architecture` string approach. If that fails, we fall back to the `sae=BatchTopKTrainingSAEConfig(...)` approach. Both are supported in sae_lens 6.37.6 — the exact interface should be verified by checking `LanguageModelSAERunnerConfig.__init__` on the VM before running.

---

## Evaluation Pipeline

After training, evaluate the SAE before integrating it into the web UI.

### Step 1: Basic quality metrics

Run `scripts/00_explore_sae.py` (already written) against the new SAE to get:
- Dead feature count (target: <10%, ideally <5%)
- L0 distribution (target: mean ~64 with natural variance, not constant)
- Decoder norm statistics

### Step 2: Feature discovery on code properties

Re-run the existing feature discovery pipeline with the new SAE:
1. `scripts/typing/01_collect_activations.py` — collect activations for typed/untyped code pairs
2. `scripts/typing/02_find_typing_features.py` — find differential features
3. `scripts/typing/04_verify_steering.py` — verify steering actually produces type annotations

### Step 3: Quantitative steering test

For each target property, measure:
- **Monotonicity**: Does increasing strength monotonically increase the target metric?
- **Coherence**: Does the output remain coherent at the working strength?
- **Specificity**: Does the feature affect only the target property, not unrelated properties?

Success criteria: At least one feature per property shows monotonic increase in the target metric while maintaining coherent output.

### Step 4: Integration test

Replace the community SAE in `backend/server.py` with the new SAE and verify the web UI works end-to-end.

---

## Execution Steps

### Pre-training setup (30 min)

```bash
# SSH to VM
ssh -i ~/.ssh/id_rsa azureuser@20.38.0.252

# Activate environment
conda activate steering

# 1. Login to wandb (for experiment tracking)
wandb login

# 3. Verify GPU 1 is free
nvidia-smi

# 4. Quick API check — verify BatchTopK is available
python -c "from sae_lens import LanguageModelSAERunnerConfig; print('OK')"

# 5. Copy training script
scp scripts/train_sae.py azureuser@20.38.0.252:~/train_sae.py
```

### Training run (3-4 hours for 2B tokens)

```bash
# Start training in a screen session (persists after SSH disconnect)
screen -S sae_training

conda activate steering
cd ~

# Initial short run to verify everything works (50M tokens, ~5 min)
python train_sae.py --tokens 50_000_000 --checkpoint-path checkpoints/code_sae_test

# If successful, start the full run
python train_sae.py --tokens 2_000_000_000 --checkpoint-path checkpoints/code_sae_v1

# Detach screen: Ctrl-A, D
```

### Monitoring

```bash
# Check training progress
screen -r sae_training

# Watch GPU utilization
watch -n 5 nvidia-smi

# Check wandb dashboard for loss curves
# URL will be printed when training starts
```

### Post-training evaluation (1-2 hours)

```bash
# 1. Profile the new SAE
python 00_explore_sae.py --sae-path checkpoints/code_sae_v1/final

# 2. Run feature discovery
cd scripts/typing
python 01_collect_activations.py --sae-path ~/checkpoints/code_sae_v1/final
python 02_find_typing_features.py
python 04_verify_steering.py

# 3. If features verify successfully, update the backend
# Change SAE_ID in backend/server.py to point to the new SAE
```

---

## Integration with Backend

Once the SAE is trained and evaluated, update `backend/server.py`:

```python
# Old
SAE_ID = "tylercosgrove/mistral-7b-sparse-autoencoder-layer16"
# ...
sae = SAE.from_pretrained(release=SAE_ID, sae_id=".")[0]

# New — load from local checkpoint
SAE_PATH = "/home/azureuser/checkpoints/code_sae_v1/final"
# ...
sae = SAE.load_from_pretrained(SAE_PATH)
# Hook point will be blocks.16.hook_resid_post (set during training)
```

The rest of the backend code (steering hook, generation) remains the same — the interface is identical.

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| Dataset access denied | Fall back to `codeparrot/github-code` (ungated) or `monology/pile-uncopyrighted` |
| OOM on GPU 1 | Reduce `train_batch_size_tokens` to 2048 or `context_size` to 256 |
| Training diverges | Lower `lr` to `1e-5`, increase `lr_warm_up_steps` to 2000 |
| Still too many dead features | Increase `aux_loss_coefficient` to 1/16 or 1/8 |
| Features don't steer | Try layer 20 (`blocks.20.hook_resid_post`) — closer to output, may have more generation-causal features |
| sae_lens API mismatch | Check exact API signatures on VM before running; the `architecture` string approach vs `BatchTopKTrainingSAEConfig` may differ between versions |
| Disk space for checkpoints | Each checkpoint ~260MB (32K × 4096 × 4 bytes × 2 matrices). 5 checkpoints = ~1.3 GB. Well within 216 GB free. |

---

## Expected Improvements Over Community SAE

| Metric | Community SAE | Expected (New) |
|---|---|---|
| Dead features | 80% | <10% |
| L0 sparsity | 128 (constant) | ~64 (natural variance) |
| Hook point | MLP output | Residual stream |
| Dictionary size | 131,072 (32x) | 32,768 (8x) |
| Context size | 256 | 512 |
| Training data | General (Pile) | Code-heavy (StarCoder/GitHub) |
| Architecture | TopK | BatchTopK |
| Steering effect on typing | None (0.000 density at all strengths) | Measurable monotonic increase |

---

## Timeline

| Phase | Duration | What |
|---|---|---|
| Setup | 30 min | HuggingFace login, wandb login, verify API, copy script |
| Test run | 10 min | 50M tokens to verify pipeline works |
| Full training | 3-4 hours | 2B tokens on GPU 1 |
| Evaluation | 1-2 hours | Profile SAE, run feature discovery, verify steering |
| Integration | 30 min | Update backend, test web UI |
| **Total** | **~6 hours** | End-to-end from setup to working web UI |

If the 2B-token run produces good results (low dead features, monotonic steering), we ship it. If marginal, extend to 4B tokens (additional 3-4 hours).
