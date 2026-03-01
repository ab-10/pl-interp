#!/usr/bin/env bash
# Stages 4-7: SAE training, feature analysis, steering experiments, and results analysis.
# Run from repo root AFTER stages 1-3 (generation, evaluation, activation capture) complete.
#
# Usage: nohup bash experiments/scripts/run_stages_4_7.sh > /scratch/stages_4_7.log 2>&1 &
#
# The model is expected at ~/models/mistral-7b-instruct-v0.3 on the VM.
# Set MODEL_PATH env var to override.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"

echo "$(date): Starting stages 4-7"
echo "Generations: $(ls /scratch/generations/shard_*.jsonl 2>/dev/null | wc -l) shards"
echo ""

# --- Stage 4: Train SAE (~30 min) ---
echo "$(date): === Stage 4: Train SAE ==="
python3 -m experiments.sae.train \
  --generations-dir /scratch/generations \
  --activations-dir /scratch/activations \
  --output-dir /scratch/sae \
  --device cuda
echo "$(date): Stage 4 complete"
echo ""

# --- Stage 5a: Analyze SAE features (~10 min) ---
echo "$(date): === Stage 5a: Analyze SAE features ==="
python3 -m experiments.sae.analyze \
  --sae-checkpoint /scratch/sae/sae_checkpoint.pt \
  --generations-dir /scratch/generations \
  --activations-dir /scratch/activations \
  --output-dir /scratch/analysis \
  --device cuda
echo "$(date): Stage 5a complete"
echo ""

# --- Stage 5b: Select steering candidates ---
echo "$(date): === Stage 5b: Select steering candidates ==="
python3 -m experiments.sae.select_candidates \
  --feature-stats /scratch/analysis/feature_stats.json \
  --sae-checkpoint /scratch/sae/sae_checkpoint.pt \
  --output-dir /scratch/analysis
echo "$(date): Stage 5b complete"
echo ""

# --- Stage 5c: Compute contrastive directions (CPU, fast) ---
echo "$(date): === Stage 5c: Compute contrastive directions ==="
python3 -m experiments.contrastive.compute_directions \
  --generations-dir /scratch/generations \
  --activations-dir /scratch/activations \
  --output-dir /scratch/analysis
echo "$(date): Stage 5c complete"
echo ""

# --- Stage 6: SAE steering (2 GPUs in parallel, ~30-45 min) ---
echo "$(date): === Stage 6: SAE steering ==="
CUDA_VISIBLE_DEVICES=0 python3 -m experiments.steering.run_experiment \
  --directions /scratch/analysis/steering_directions.pt \
  --experiment-name sae_steering \
  --output-dir /scratch/steering \
  --shard 0 --num-shards 2 --include-random-controls &
PID_SAE0=$!

CUDA_VISIBLE_DEVICES=1 python3 -m experiments.steering.run_experiment \
  --directions /scratch/analysis/steering_directions.pt \
  --experiment-name sae_steering \
  --output-dir /scratch/steering \
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

# --- Stage 7: Contrastive steering (2 GPUs in parallel, ~1-1.5h) ---
echo "$(date): === Stage 7: Contrastive steering ==="
CUDA_VISIBLE_DEVICES=0 python3 -m experiments.steering.run_experiment \
  --directions /scratch/analysis/contrastive_directions.pt \
  --experiment-name contrastive_steering \
  --output-dir /scratch/steering \
  --shard 0 --num-shards 2 &
PID_CON0=$!

CUDA_VISIBLE_DEVICES=1 python3 -m experiments.steering.run_experiment \
  --directions /scratch/analysis/contrastive_directions.pt \
  --experiment-name contrastive_steering \
  --output-dir /scratch/steering \
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
echo ""

# --- Stage 8: Analyze all results ---
echo "$(date): === Stage 8: Analyze results ==="
python3 -m experiments.steering.analyze_steering \
  --steering-dir /scratch/steering \
  --output-dir /scratch/analysis
echo "$(date): Stage 8 complete"
echo ""

# --- Copy to persistent storage ---
echo "$(date): === Copying results to persistent storage ==="
mkdir -p ~/results
rsync -av /scratch/generations/ ~/results/generations/
rsync -av /scratch/sae/ ~/results/sae/
rsync -av /scratch/steering/ ~/results/steering/
rsync -av /scratch/analysis/ ~/results/analysis/
echo "$(date): Copy complete"
echo ""

echo "$(date): === ALL STAGES 4-7 COMPLETE ==="
