#!/usr/bin/env bash
# Stages 4-7: SAE training, feature analysis, steering experiments, and results analysis.
# Run from repo root AFTER stages 1-3 (generation, evaluation, activation capture) complete.
#
# Usage: nohup bash experiments/scripts/run_stages_4_7.sh --model ministral-8b > /scratch/ministral-8b/stages_4_7.log 2>&1 &
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"

# --- Parse --model argument (same pattern as run_pipeline.sh) ---
MODEL="${1:---model}"
MODEL_NAME="${2:-ministral-8b}"
if [[ "$MODEL" != "--model" ]]; then
    MODEL_NAME="$MODEL"
fi

# Model-specific scratch paths
S="/scratch/$MODEL_NAME"
GEN_DIR="$S/generations"
ACT_DIR="$S/activations"
SAE_DIR="$S/sae"
ANALYSIS_DIR="$S/analysis"
STEERING_DIR="$S/steering"
mkdir -p "$SAE_DIR" "$ANALYSIS_DIR" "$STEERING_DIR"

echo "$(date): Starting stages 4-7 for model=$MODEL_NAME"
echo "Scratch root: $S"
echo "Generations: $(ls "$GEN_DIR"/shard_*.jsonl 2>/dev/null | wc -l) shards"
echo ""

# --- Ensure wandb is installed ---
if ! python3 -c "import wandb" 2>/dev/null; then
    echo "Installing wandb..."
    pip install wandb -q
fi
echo "wandb: $(python3 -c 'import wandb; print(wandb.__version__)')"
echo ""

# --- Symlink for activation_file paths in generation records ---
# Generation records store absolute paths like /scratch/activations/shard_0.npy
# (from older code). Create symlink so those paths resolve correctly.
if [ ! -e /scratch/activations ] && [ -d "$ACT_DIR" ]; then
    ln -sfn "$ACT_DIR" /scratch/activations
    echo "Created symlink /scratch/activations -> $ACT_DIR"
fi

# --- Stage 4: Train SAE (~30 min) ---
echo "$(date): === Stage 4: Train SAE ==="
python3 -m experiments.sae.train \
  --generations-dir "$GEN_DIR" \
  --activations-dir "$ACT_DIR" \
  --output-dir "$SAE_DIR" \
  --device cuda
echo "$(date): Stage 4 complete"
echo ""

# --- Stage 5a: Analyze SAE features (~10 min) ---
echo "$(date): === Stage 5a: Analyze SAE features ==="
python3 -m experiments.sae.analyze \
  --sae-checkpoint "$SAE_DIR/sae_checkpoint.pt" \
  --generations-dir "$GEN_DIR" \
  --activations-dir "$ACT_DIR" \
  --output-dir "$ANALYSIS_DIR" \
  --device cuda
echo "$(date): Stage 5a complete"
echo ""

# --- Stage 5b: Select steering candidates ---
echo "$(date): === Stage 5b: Select steering candidates ==="
python3 -m experiments.sae.select_candidates \
  --feature-stats "$ANALYSIS_DIR/feature_stats.json" \
  --sae-checkpoint "$SAE_DIR/sae_checkpoint.pt" \
  --output-dir "$ANALYSIS_DIR"
echo "$(date): Stage 5b complete"
echo ""

# --- Stage 5c: Compute contrastive directions (CPU, fast) ---
echo "$(date): === Stage 5c: Compute contrastive directions ==="
python3 -m experiments.contrastive.compute_directions \
  --generations-dir "$GEN_DIR" \
  --activations-dir "$ACT_DIR" \
  --output-dir "$ANALYSIS_DIR"
echo "$(date): Stage 5c complete"
echo ""

# --- Stage 6: SAE steering (2 GPUs in parallel, ~30-45 min) ---
echo "$(date): === Stage 6: SAE steering ==="
CUDA_VISIBLE_DEVICES=0 python3 -m experiments.steering.run_experiment \
  --directions "$ANALYSIS_DIR/steering_directions.pt" \
  --experiment-name sae_steering \
  --output-dir "$STEERING_DIR" \
  --shard 0 --num-shards 2 --include-random-controls &
PID_SAE0=$!

CUDA_VISIBLE_DEVICES=1 python3 -m experiments.steering.run_experiment \
  --directions "$ANALYSIS_DIR/steering_directions.pt" \
  --experiment-name sae_steering \
  --output-dir "$STEERING_DIR" \
  --shard 1 --num-shards 2 --include-random-controls &
PID_SAE1=$!

echo "  SAE shard 0 PID=$PID_SAE0, shard 1 PID=$PID_SAE1"

FAIL=0
wait $PID_SAE0 || FAIL=1
wait $PID_SAE1 || FAIL=1

if [ $FAIL -ne 0 ]; then
    echo "FATAL: SAE steering failed. Check logs above."
    exit 1
fi
echo "$(date): Stage 6 complete"
echo ""

# --- Stage 7: Contrastive steering (SKIPPED by default, pass --contrastive to enable) ---
if [ "${RUN_CONTRASTIVE:-0}" = "1" ]; then
    echo "$(date): === Stage 7: Contrastive steering ==="
    CUDA_VISIBLE_DEVICES=0 python3 -m experiments.steering.run_experiment \
      --directions "$ANALYSIS_DIR/contrastive_directions.pt" \
      --experiment-name contrastive_steering \
      --output-dir "$STEERING_DIR" \
      --shard 0 --num-shards 2 &
    PID_CON0=$!

    CUDA_VISIBLE_DEVICES=1 python3 -m experiments.steering.run_experiment \
      --directions "$ANALYSIS_DIR/contrastive_directions.pt" \
      --experiment-name contrastive_steering \
      --output-dir "$STEERING_DIR" \
      --shard 1 --num-shards 2 &
    PID_CON1=$!

    echo "  Contrastive shard 0 PID=$PID_CON0, shard 1 PID=$PID_CON1"

    FAIL=0
    wait $PID_CON0 || FAIL=1
    wait $PID_CON1 || FAIL=1

    if [ $FAIL -ne 0 ]; then
        echo "FATAL: Contrastive steering failed. Check logs above."
        exit 1
    fi
    echo "$(date): Stage 7 complete"
else
    echo "$(date): === Stage 7: Contrastive steering SKIPPED (set RUN_CONTRASTIVE=1 to enable) ==="
fi
echo ""

# --- Stage 8: Analyze all results ---
echo "$(date): === Stage 8: Analyze results ==="
python3 -m experiments.steering.analyze_steering \
  --steering-dir "$STEERING_DIR" \
  --output-dir "$ANALYSIS_DIR"
echo "$(date): Stage 8 complete"
echo ""

# --- Copy to persistent storage ---
echo "$(date): === Copying results to persistent storage ==="
mkdir -p ~/results/"$MODEL_NAME"
rsync -av "$GEN_DIR/" ~/results/"$MODEL_NAME"/generations/
rsync -av "$SAE_DIR/" ~/results/"$MODEL_NAME"/sae/
rsync -av "$STEERING_DIR/" ~/results/"$MODEL_NAME"/steering/
rsync -av "$ANALYSIS_DIR/" ~/results/"$MODEL_NAME"/analysis/
echo "$(date): Copy complete"
echo ""

echo "$(date): === ALL STAGES 4-7 COMPLETE ==="
