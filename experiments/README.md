# Experiments Pipeline

Code generation + mechanistic interpretability pipeline. Multi-model support
(Mistral-7B, Ministral-8B). Generates code across 6 prompt variants, captures
activations at configurable layers, trains TopK SAEs, discovers steering features,
and runs steering experiments.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (tested on 2x H100 80GB)
- Dependencies: `pip install -r requirements.txt`

## Quick Start

```bash
cd /path/to/pl-interp

# 1. Install dependencies
pip install -r experiments/requirements.txt

# 2. Create scratch directories (model-specific)
mkdir -p /scratch/ministral-8b/{generations,activations,sae,steering,analysis}

# 3. Run smoke test (validates GPU, indexing, hooks, token round-trips)
python -m experiments.scripts.00_sanity_check

# 4. (Optional) Run full E2E smoke test with actual vLLM + activation capture
python -m experiments.scripts.00_sanity_check --full-e2e
```

## Model Configuration

Config lives in `experiments/config.py`. Models are registered as presets:

```bash
--model mistral-7b     # Mistral-7B-Instruct-v0.3, layers 16/24
--model ministral-8b   # Ministral-8B-Instruct-2410, layers 18/27
```

Capture layers are auto-computed at 50% and 75% of model depth.

## Script Execution Order

All scripts use `argparse` — run with `--help` for full options.
Run from the repo root (`pl-interp/`).

```bash
# 0. Validate GPU setup, indexing, and hooks (~2 min)
python -m experiments.scripts.00_sanity_check

# 1. Generate all code outputs with vLLM (~20 min, 2 GPU shards)
CUDA_VISIBLE_DEVICES=0 python -m experiments.scripts.01_generate --shard 0 &
CUDA_VISIBLE_DEVICES=1 python -m experiments.scripts.01_generate --shard 1 &
wait

# 2. Run test execution on all outputs (~30 min, CPU-bound)
python -m experiments.scripts.02_evaluate

# 3. Capture activations via HF teacher-forcing (~30 min, 2 GPU shards)
CUDA_VISIBLE_DEVICES=0 python -m experiments.scripts.03_capture_activations --shard 0 &
CUDA_VISIBLE_DEVICES=1 python -m experiments.scripts.03_capture_activations --shard 1 &
wait

# 4-7. SAE training, feature analysis, steering (via run script)
bash experiments/scripts/run_stages_4_7.sh --model ministral-8b

# 8. Label top features with Claude on Bedrock (~3 min per 50 features)
python -m experiments.sae.label_features --model ministral-8b --layer 18 --top-n 50

# 9. Analyze feature success (LLM judges contribution to correctness)
python -m experiments.sae.analyze_success --model ministral-8b --layer 18
```

### Stages 4-7 Detail (`run_stages_4_7.sh`)

4. **SAE Training** — TopK SAE (K=64) on captured activations
5. **Feature Analysis** — Cohen's d, variant means, probe training, candidate selection
6. **Steering Experiment** — Generate with each feature direction at multiple alphas
7. **Steering Analysis** — Property density, monotonicity, contamination checks

## Smoke Test Modes

```bash
# Default: 5 sanity checks + micro E2E (uses HF generate, ~2 min)
python -m experiments.scripts.00_sanity_check

# Sanity checks only (skip all E2E tests)
python -m experiments.scripts.00_sanity_check --skip-e2e

# Full pipeline E2E: vLLM generate → extract → execute → capture activations (~5 min)
python -m experiments.scripts.00_sanity_check --full-e2e
```

The 5 sanity checks validate:
1. Hidden state indexing (correct layer offset)
2. Shape validation (hidden dim, hidden states count)
3. Teacher-forcing determinism (bitwise identical on repeated forward pass)
4. Steering hook (alpha=0 no-op, large alpha differs, decode-only gating)
5. Token ID round-trip (decode → re-encode preserves IDs)

## Storage Layout (`/scratch`)

All intermediate data lives on the VM's NVMe scratch disk, namespaced by model:

```
/scratch/ministral-8b/
  generations/        JSONL shards (shard_0.jsonl, shard_1.jsonl)
  activations/        float16 mmap files (shard_0.npy, shard_1.npy)
  sae/
    layer_18/         SAE checkpoint + training log
    layer_27/         SAE checkpoint + training log
  analysis/
    layer_18/         feature_stats.json, feature_candidates.json, steering_directions.pt,
                      steering_results.json, feature_labels.json, feature_success_analysis.json,
                      probe_direction.pt, probe_stats.json
    layer_27/         (same structure)
  steering/           Steering results JSONL
```

Scratch is **ephemeral** — lost on VM deallocation. After the pipeline
completes, copy results to the persistent OS disk:

```bash
rsync -av /scratch/ministral-8b/ ~/results/ministral-8b/
# Skip raw activations (~50GB) — regenerable from stored token IDs
```

## GPU Sharding

Generation and activation capture shard by task ID across GPUs. Each shard
processes roughly half the tasks independently. Use `CUDA_VISIBLE_DEVICES` to
pin each process to one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.scripts.01_generate --shard 0
CUDA_VISIBLE_DEVICES=1 python -m experiments.scripts.01_generate --shard 1
```

## Resuming After Failure

- **Generation (01)**: Delete the shard JSONL and re-run. Generation is fast enough that partial resume isn't worth the complexity.
- **Evaluation (02)**: Safe to re-run — overwrites shard files atomically (tmp + rename).
- **Activation capture (03)**: Skips records that already have `activation_file` set. Safe to re-run after a crash — the mmap writer tracks row offset from file size.
