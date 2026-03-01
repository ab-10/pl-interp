#!/usr/bin/env bash
set -euo pipefail

# Run the Ministral 8B feature steering server.
#
# Required env vars:
#   SAE_CHECKPOINT       — path to sae_checkpoint.pt
#
# Optional:
#   MODEL_PATH     — HF model ID or local path (default: ~/models/Ministral-8B-Instruct-2410)
#   STEER_LAYER    — layer index (default: 18)
#   MAX_NEW_TOKENS — generation length (default: 200)
#
# Usage:
#   SAE_CHECKPOINT=~/8b_saes/layer_18_sae_checkpoint.pt \
#   MODEL_PATH=~/models/Ministral-8B-Instruct-2410 \
#   bash backend/run_server.sh

export MODEL_PATH="${MODEL_PATH:-$HOME/models/Ministral-8B-Instruct-2410}"
export PATH="$HOME/.local/bin:$PATH"

echo "=== Feature Steering Server ==="
echo "  SAE checkpoint:      ${SAE_CHECKPOINT}"
echo "  Model:               ${MODEL_PATH}"
echo "  Steer layer:         ${STEER_LAYER:-18}"
echo ""

cd "$(dirname "$0")/.."
uvicorn backend.server:app --host 0.0.0.0 --port 8000
