#!/usr/bin/env bash
set -euo pipefail

# Run the custom SAE steering server.
#
# Required env vars:
#   SAE_CHECKPOINT       — path to sae_checkpoint.pt
#   FEATURE_CANDIDATES   — path to feature_candidates.json
#
# Optional:
#   STEERING_RESULTS — path to steering_results.json (enriches feature labels)
#   MODEL_PATH       — HF model ID or local path (default: mistralai/Mistral-7B-Instruct-v0.3)
#   STEER_LAYER      — layer index (default: 16)
#   MAX_NEW_TOKENS   — generation length (default: 200)
#
# Examples:
#   # Mistral 7B
#   SAE_CHECKPOINT=artifacts/mistral-7b/sae_checkpoint.pt \
#   FEATURE_CANDIDATES=artifacts/mistral-7b/feature_candidates.json \
#   STEERING_RESULTS=artifacts/mistral-7b/steering_results.json \
#   STEER_LAYER=16 \
#   bash backend/run_custom_sae_server.sh
#
#   # Ministral 8B (layer 18)
#   SAE_CHECKPOINT=artifacts/ministral-8b/layer_18/sae_checkpoint.pt \
#   FEATURE_CANDIDATES=artifacts/ministral-8b/layer_18/feature_candidates.json \
#   STEERING_RESULTS=artifacts/ministral-8b/layer_18/steering_results.json \
#   MODEL_PATH=mistralai/Ministral-8B-Instruct-2410 \
#   STEER_LAYER=18 \
#   bash backend/run_custom_sae_server.sh

export MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-7B-Instruct-v0.3}"
export PATH="$HOME/.local/bin:$PATH"

echo "=== Custom SAE Steering Server ==="
echo "  SAE checkpoint:      ${SAE_CHECKPOINT}"
echo "  Feature candidates:  ${FEATURE_CANDIDATES}"
echo "  Steering results:    ${STEERING_RESULTS:-<not set>}"
echo "  Model:               ${MODEL_PATH}"
echo "  Steer layer:         ${STEER_LAYER:-16}"
echo ""

cd "$(dirname "$0")/.."
uvicorn backend.server_custom_sae:app --host 0.0.0.0 --port 8000
