#!/bin/bash
# Full pipeline launcher. Runs all 3 stages with 2 GPU shards.
# Usage: nohup bash experiments/scripts/run_pipeline.sh > /scratch/pipeline.log 2>&1 &
set -e

cd ~/pl-interp
export PATH=$HOME/.local/bin:$PATH

echo "=== Pipeline started at $(date) ==="
echo ""

# --- Stage 1: Generate (2 GPUs in parallel) ---
echo "=== Stage 1: Generation ==="
CUDA_VISIBLE_DEVICES=0 python3 -m experiments.scripts.01_generate --shard 0 --num-shards 2 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 python3 -m experiments.scripts.01_generate --shard 1 --num-shards 2 &
PID1=$!

echo "  Shard 0 PID=$PID0, Shard 1 PID=$PID1"
echo "  Waiting for both generation shards..."

FAIL=0
wait $PID0 || FAIL=1
wait $PID1 || FAIL=1

if [ $FAIL -ne 0 ]; then
    echo "FATAL: Generation failed. Check logs above."
    exit 1
fi
echo "=== Stage 1 complete at $(date) ==="
echo ""

# --- Stage 2: Evaluate (CPU only, single process) ---
echo "=== Stage 2: Evaluation ==="
python3 -m experiments.scripts.02_evaluate
echo "=== Stage 2 complete at $(date) ==="
echo ""

# --- Stage 3: Capture activations (2 GPUs in parallel) ---
echo "=== Stage 3: Activation Capture ==="
CUDA_VISIBLE_DEVICES=0 python3 -m experiments.scripts.03_capture_activations --shard 0 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 python3 -m experiments.scripts.03_capture_activations --shard 1 &
PID1=$!

echo "  Shard 0 PID=$PID0, Shard 1 PID=$PID1"
echo "  Waiting for both capture shards..."

FAIL=0
wait $PID0 || FAIL=1
wait $PID1 || FAIL=1

if [ $FAIL -ne 0 ]; then
    echo "FATAL: Activation capture failed. Check logs above."
    exit 1
fi
echo "=== Stage 3 complete at $(date) ==="
echo ""

echo "=== Pipeline finished at $(date) ==="
echo "Check /scratch/generations/ and /scratch/activations/"
