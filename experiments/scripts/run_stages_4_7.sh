#!/usr/bin/env bash
# Stages 5-8: Feature analysis, steering experiments, and results analysis.
# Runs per-layer: for each capture layer, analyzes SAE features, selects
# steering candidates, computes contrastive directions, runs steering on
# HumanEval, and analyzes results.
#
# Prerequisite: SAEs already trained (stage 4) at /scratch/<model>/sae/layer_<N>/sae_checkpoint.pt
#
# Usage: nohup bash experiments/scripts/run_stages_4_7.sh --model ministral-8b > /scratch/ministral-8b/stages_5_8.log 2>&1 &
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"

# --- Parse --model argument ---
MODEL="${1:---model}"
MODEL_NAME="${2:-ministral-8b}"
if [[ "$MODEL" != "--model" ]]; then
    MODEL_NAME="$MODEL"
fi

# Model-specific scratch paths
S="/scratch/$MODEL_NAME"
GEN_DIR="$S/generations"
ACT_DIR="$S/activations"

echo "$(date): Starting stages 5-8 for model=$MODEL_NAME"
echo "Scratch root: $S"
echo "Generations: $(ls "$GEN_DIR"/shard_*.jsonl 2>/dev/null | wc -l) shards"

# Discover capture layers from existing SAE checkpoints
LAYERS=()
for layer_dir in "$S"/sae/layer_*/; do
    if [ -f "${layer_dir}sae_checkpoint.pt" ]; then
        layer_num=$(basename "$layer_dir" | sed 's/layer_//')
        LAYERS+=("$layer_num")
    fi
done

if [ ${#LAYERS[@]} -eq 0 ]; then
    echo "FATAL: No SAE checkpoints found in $S/sae/layer_*/sae_checkpoint.pt"
    exit 1
fi
echo "Layers with trained SAEs: ${LAYERS[*]}"
echo ""

# --- Ensure wandb is installed ---
if ! python3 -c "import wandb" 2>/dev/null; then
    echo "Installing wandb..."
    pip install wandb -q
fi

# =====================================================================
# Per-layer pipeline: stages 5a, 5b, 5c for each layer
# =====================================================================
for LAYER in "${LAYERS[@]}"; do
    SAE_DIR="$S/sae/layer_$LAYER"
    ANALYSIS_DIR="$S/analysis/layer_$LAYER"
    STEERING_DIR="$S/steering/layer_$LAYER"
    mkdir -p "$ANALYSIS_DIR" "$STEERING_DIR"

    echo ""
    echo "$(date): =============================================="
    echo "$(date): Processing layer $LAYER"
    echo "$(date): =============================================="

    # --- Stage 5a: Analyze SAE features ---
    echo "$(date): === Stage 5a: Analyze SAE features (layer $LAYER) ==="
    python3 -m experiments.sae.analyze \
      --sae-checkpoint "$SAE_DIR/sae_checkpoint.pt" \
      --generations-dir "$GEN_DIR" \
      --activations-dir "$ACT_DIR" \
      --output-dir "$ANALYSIS_DIR" \
      --layer "$LAYER" \
      --device cuda
    echo "$(date): Stage 5a complete (layer $LAYER)"
    echo ""

    # --- Stage 5b: Select steering candidates ---
    echo "$(date): === Stage 5b: Select steering candidates (layer $LAYER) ==="
    python3 -m experiments.sae.select_candidates \
      --feature-stats "$ANALYSIS_DIR/feature_stats.json" \
      --sae-checkpoint "$SAE_DIR/sae_checkpoint.pt" \
      --output-dir "$ANALYSIS_DIR"
    echo "$(date): Stage 5b complete (layer $LAYER)"
    echo ""

    # --- Stage 5c: Compute contrastive directions ---
    echo "$(date): === Stage 5c: Compute contrastive directions (layer $LAYER) ==="
    python3 -m experiments.contrastive.compute_directions \
      --generations-dir "$GEN_DIR" \
      --activations-dir "$ACT_DIR" \
      --output-dir "$ANALYSIS_DIR" \
      --layer "$LAYER"
    echo "$(date): Stage 5c complete (layer $LAYER)"
    echo ""

    # --- Stage 5d: Label features with LLM ---
    echo "$(date): === Stage 5d: Label features with LLM (layer $LAYER) ==="
    python3 -m experiments.sae.label_features \
      --model "$MODEL_NAME" \
      --layer "$LAYER" \
      --top-n 50
    echo "$(date): Stage 5d complete (layer $LAYER)"
    echo ""
done

# =====================================================================
# Stage 6: SAE steering — both layers in parallel (2 GPUs)
# =====================================================================
echo "$(date): === Stage 6: SAE steering (all layers, 2 GPUs) ==="

# We run one layer per GPU for parallel execution.
# Each layer uses 1 shard (no task splitting since we have 1 GPU per layer).
PIDS=()
GPU=0
for LAYER in "${LAYERS[@]}"; do
    ANALYSIS_DIR="$S/analysis/layer_$LAYER"
    STEERING_DIR="$S/steering/layer_$LAYER"

    CUDA_VISIBLE_DEVICES=$GPU python3 -m experiments.steering.run_experiment \
      --directions "$ANALYSIS_DIR/steering_directions.pt" \
      --experiment-name sae_steering \
      --output-dir "$STEERING_DIR" \
      --steer-layer "$LAYER" \
      --shard 0 --num-shards 1 --include-random-controls &
    PIDS+=($!)
    echo "  Layer $LAYER on GPU $GPU, PID=${PIDS[-1]}"
    GPU=$((GPU + 1))
done

FAIL=0
for PID in "${PIDS[@]}"; do
    wait $PID || FAIL=1
done

if [ $FAIL -ne 0 ]; then
    echo "FATAL: SAE steering failed. Check logs above."
    exit 1
fi
echo "$(date): Stage 6 complete"
echo ""

# =====================================================================
# Stage 8: Analyze results per layer
# =====================================================================
for LAYER in "${LAYERS[@]}"; do
    ANALYSIS_DIR="$S/analysis/layer_$LAYER"
    STEERING_DIR="$S/steering/layer_$LAYER"

    echo "$(date): === Stage 8: Analyze results (layer $LAYER) ==="
    python3 -m experiments.steering.analyze_steering \
      --steering-dir "$STEERING_DIR" \
      --output-dir "$ANALYSIS_DIR"
    echo "$(date): Stage 8 complete (layer $LAYER)"
    echo ""
done

# --- Copy to persistent storage ---
echo "$(date): === Copying results to persistent storage ==="
mkdir -p ~/results/"$MODEL_NAME"
rsync -av "$S/sae/" ~/results/"$MODEL_NAME"/sae/
rsync -av "$S/steering/" ~/results/"$MODEL_NAME"/steering/
rsync -av "$S/analysis/" ~/results/"$MODEL_NAME"/analysis/
echo "$(date): Copy complete"
echo ""

echo "$(date): === ALL STAGES 5-8 COMPLETE ==="
