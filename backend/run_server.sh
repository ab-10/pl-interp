#!/usr/bin/env bash
set -euo pipefail

# Activate the steering conda environment and start the server
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate steering

pip install -q fastapi 'uvicorn[standard]'

echo "Starting Feature Steering API on port 8000..."
uvicorn server:app --host 0.0.0.0 --port 8000
