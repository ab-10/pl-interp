#!/usr/bin/env bash
# Train a pass/fail probe on SAE features, then steer with the resulting direction.
# Runs per-layer: for each layer with a trained SAE, trains probe and steers.
#
# Usage: nohup bash experiments/scripts/run_probe_steer.sh --model ministral-8b > /scratch/ministral-8b/probe_steer.log 2>&1 &
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

S="/scratch/$MODEL_NAME"
GEN_DIR="$S/generations"
ACT_DIR="$S/activations"

echo "$(date): Starting probe steering for model=$MODEL_NAME"

# Discover layers with trained SAEs
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

# =====================================================================
# Per-layer: train probe, steer, analyze
# =====================================================================
for LAYER in "${LAYERS[@]}"; do
    SAE_DIR="$S/sae/layer_$LAYER"
    ANALYSIS_DIR="$S/analysis/layer_$LAYER"
    STEERING_DIR="$S/steering/layer_$LAYER"
    mkdir -p "$ANALYSIS_DIR" "$STEERING_DIR"

    echo ""
    echo "$(date): =============================================="
    echo "$(date): Probe steering — layer $LAYER"
    echo "$(date): =============================================="

    # --- Train probe ---
    echo "$(date): === Training probe (layer $LAYER) ==="
    python3 -m experiments.sae.probe \
      --sae-checkpoint "$SAE_DIR/sae_checkpoint.pt" \
      --generations-dir "$GEN_DIR" \
      --activations-dir "$ACT_DIR" \
      --output-dir "$ANALYSIS_DIR" \
      --layer "$LAYER" \
      --device cuda
    echo "$(date): Probe training complete (layer $LAYER)"

    # --- Steer with probe direction (2 GPUs) ---
    echo "$(date): === Steering with probe direction (layer $LAYER) ==="
    CUDA_VISIBLE_DEVICES=0 python3 -m experiments.steering.run_experiment \
      --directions "$ANALYSIS_DIR/probe_direction.pt" \
      --experiment-name probe_steering \
      --output-dir "$STEERING_DIR" \
      --steer-layer "$LAYER" \
      --shard 0 --num-shards 2 &
    PID0=$!
    CUDA_VISIBLE_DEVICES=1 python3 -m experiments.steering.run_experiment \
      --directions "$ANALYSIS_DIR/probe_direction.pt" \
      --experiment-name probe_steering \
      --output-dir "$STEERING_DIR" \
      --steer-layer "$LAYER" \
      --shard 1 --num-shards 2 &
    PID1=$!
    echo "  Shard PIDs: $PID0, $PID1"
    FAIL=0; wait $PID0 || FAIL=1; wait $PID1 || FAIL=1
    if [ $FAIL -ne 0 ]; then echo "FATAL: Probe steering failed (layer $LAYER)"; exit 1; fi
    echo "$(date): Steering complete (layer $LAYER)"

    # --- Analyze ---
    echo "$(date): === Analyzing (layer $LAYER) ==="
    python3 -m experiments.steering.analyze_steering \
      --steering-dir "$STEERING_DIR" \
      --output-dir "$ANALYSIS_DIR"
    echo "$(date): Analysis complete (layer $LAYER)"
done

# --- Copy results ---
echo ""
echo "$(date): === Copying results ==="
mkdir -p ~/results/"$MODEL_NAME"
rsync -av "$S/steering/" ~/results/"$MODEL_NAME"/steering/
rsync -av "$S/analysis/" ~/results/"$MODEL_NAME"/analysis/

echo "$(date): ALL DONE"
