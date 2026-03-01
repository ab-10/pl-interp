# Experiments Pipeline

Code generation + mechanistic interpretability pipeline for Mistral 7B.
Generates code across 6 prompt variants, captures layer-16 activations, trains
an SAE, and runs steering experiments.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (tested on 2x H100 80GB)
- Dependencies: `pip install -r requirements.txt`

## Quick Start

```bash
cd /path/to/pl-interp

# 1. Install dependencies
pip install -r experiments/requirements.txt

# 2. Create scratch directories
mkdir -p /scratch/{generations,activations,sae,steering,analysis}

# 3. Run smoke test (validates GPU, indexing, hooks, token round-trips)
python -m experiments.scripts.00_sanity_check

# 4. (Optional) Run full E2E smoke test with actual vLLM + activation capture
python -m experiments.scripts.00_sanity_check --full-e2e
```

## Script Execution Order

All scripts use `argparse` — run with `--help` for full options.
Run from the repo root (`pl-interp/`).

```bash
# 0. Validate GPU setup, indexing, and hooks (~2 min)
python -m experiments.scripts.00_sanity_check

# 1. Generate all code outputs with vLLM (~1h, 2 GPU shards)
CUDA_VISIBLE_DEVICES=0 python -m experiments.scripts.01_generate --shard 0 &
CUDA_VISIBLE_DEVICES=1 python -m experiments.scripts.01_generate --shard 1 &
wait

# 2. Run test execution on all outputs (~30 min, CPU-bound)
python -m experiments.scripts.02_evaluate

# 3. Capture layer-16 activations via HF teacher-forcing (~30 min, 2 GPU shards)
CUDA_VISIBLE_DEVICES=0 python -m experiments.scripts.03_capture_activations --shard 0 &
CUDA_VISIBLE_DEVICES=1 python -m experiments.scripts.03_capture_activations --shard 1 &
wait

# 4-7. SAE training, analysis, steering (scripts not yet implemented)
# python -m experiments.scripts.04_train_sae
# python -m experiments.scripts.05_sae_steering
# python -m experiments.scripts.06_contrastive  (fallback)
# python -m experiments.scripts.07_analyze_all
```

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
1. Hidden state indexing (output_hidden_states[17] == hooked layer 16)
2. Shape validation (4096 hidden dim, 33 hidden states)
3. Teacher-forcing determinism (bitwise identical on repeated forward pass)
4. Steering hook (alpha=0 no-op, large alpha differs, decode-only gating)
5. Token ID round-trip (decode → re-encode preserves IDs)

## Storage Layout (`/scratch`)

All intermediate data lives on the VM's NVMe scratch disk:

```
/scratch/
  generations/        JSONL shards (shard_0.jsonl, shard_1.jsonl)
  activations/        float16 mmap files (shard_0.npy, shard_1.npy)
  sae/                SAE checkpoint (model.pt) + training log
  steering/           Steering results JSONL
  analysis/           Final outputs (pass_rates.csv, feature_candidates.json)
```

Scratch is **ephemeral** — lost on VM deallocation. After the pipeline
completes, copy results to the persistent OS disk:

```bash
rsync -av /scratch/generations/ ~/results/generations/
rsync -av /scratch/sae/ ~/results/sae/
rsync -av /scratch/steering/ ~/results/steering/
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
