#!/usr/bin/env bash
set -euo pipefail

# Run the custom SAE steering server.
#
# Required env vars:
#   SAE_CHECKPOINT       — path to sae_checkpoint.pt
#   FEATURE_CANDIDATES   — path to feature_candidates.json
#
# Optional:
#   MODEL_PATH     — HF model ID or local path (default: ~/models/mistral-7b-instruct-v0.3)
#   STEER_LAYER    — layer index (default: 16)
#   MAX_NEW_TOKENS — generation length (default: 200)
#
# Usage:
#   SAE_CHECKPOINT=/scratch/sae/sae_checkpoint.pt \
#   FEATURE_CANDIDATES=/scratch/analysis/feature_candidates.json \
#   MODEL_PATH=~/models/mistral-7b-instruct-v0.3 \
#   bash backend/run_server.sh

export MODEL_PATH="${MODEL_PATH:-$HOME/models/mistral-7b-instruct-v0.3}"
export PATH="$HOME/.local/bin:$PATH"

echo "=== Feature Steering Server ==="
echo "  SAE checkpoint:      ${SAE_CHECKPOINT}"
echo "  Feature candidates:  ${FEATURE_CANDIDATES}"
echo "  Model:               ${MODEL_PATH}"
echo "  Steer layer:         ${STEER_LAYER:-16}"
echo ""

cd "$(dirname "$0")/.."
uvicorn backend.server:app --host 0.0.0.0 --port 8000
