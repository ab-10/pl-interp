# VM Smoke Test Instructions

Run these steps on the GPU VM to validate the experiment pipeline before committing to full runs.

## Prerequisites

```bash
# 1. Clone or copy the repo to the VM
# The experiments/ directory must be at /path/to/pl-interp/experiments/

# 2. Install Python dependencies (in your conda env or venv)
cd /path/to/pl-interp
pip install -r experiments/requirements.txt

# 3. Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# 4. Create scratch directories
mkdir -p /scratch/{generations,activations,sae,steering,analysis}
```

## Step 1: Run Sanity Checks (~2 min)

```bash
cd /path/to/pl-interp
python -m experiments.scripts.00_sanity_check
```

This runs 5 checks + a micro end-to-end test using HF generate (no vLLM needed):

1. **Hidden state indexing**: Verifies `output_hidden_states[17]` matches a `register_forward_hook` on `model.model.layers[16]`. Must be exactly equal (torch.equal). Catches off-by-one bugs.

2. **Shape validation**: Confirms `hidden_states[17].shape[-1] == 4096` and there are 33 hidden states (1 embedding + 32 layers).

3. **Teacher-forcing determinism**: Same input tokens → bitwise identical activations on two forward passes. Proves `model.eval()` + `torch.manual_seed()` gives reproducibility.

4. **Steering hook**: Tests three things:
   - alpha=0 steering hook produces output identical to unhooked generation (proves hook doesn't corrupt)
   - alpha=10.0 produces visibly different output (proves hook is active)
   - Hook counter: fires P+N times total (P=1 prefill + N decode), but only injects on N decode steps (where `hidden_states.shape[1] == 1`)

5. **Token ID round-trip**: Generate tokens, decode to text, re-encode to IDs. Must match exactly. Catches tokenizer inconsistencies.

6. **Micro end-to-end**: Loads 2 HumanEval tasks, generates code (via HF generate), extracts with regex, executes in sandbox, classifies failure, builds a full GenerationRecord, verifies JSON round-trip.

**Expected output**: All checks print `[PASS]` and script exits with code 0.

## Step 2: Run Full E2E (~5 min)

```bash
python -m experiments.scripts.00_sanity_check --full-e2e
```

This does everything in Step 1 PLUS a complete pipeline test with actual vLLM:

1. Initializes VLLMRunner (loads Mistral 7B into vLLM)
2. Generates 2 HumanEval outputs with real vLLM batch generation
3. Extracts code, executes in sandbox, classifies
4. Captures layer-16 activations via HF teacher-forcing (`output_hidden_states=True`)
5. Stores activations to a temp mmap file via ActivationWriter
6. Reads them back via ActivationReader and verifies shapes match
7. Verifies GenerationRecord JSON round-trip with all fields populated
8. Verifies vLLM token IDs survive decode → re-encode round-trip (catches vLLM/HF tokenizer mismatch)

All temp files are cleaned up automatically.

**Expected output**: All checks print `[PASS]`, script exits 0.

## Step 3: Run Full Pipeline

Only after both smoke tests pass:

```bash
# Generate (2 GPU shards, ~1h)
CUDA_VISIBLE_DEVICES=0 python -m experiments.scripts.01_generate --shard 0 &
CUDA_VISIBLE_DEVICES=1 python -m experiments.scripts.01_generate --shard 1 &
wait

# Evaluate (CPU, ~30 min)
python -m experiments.scripts.02_evaluate

# Capture activations (2 GPU shards, ~30 min)
CUDA_VISIBLE_DEVICES=0 python -m experiments.scripts.03_capture_activations --shard 0 &
CUDA_VISIBLE_DEVICES=1 python -m experiments.scripts.03_capture_activations --shard 1 &
wait
```

## Troubleshooting

- **"CUDA is not available"**: Check `nvidia-smi`, verify PyTorch was installed with CUDA support.
- **Sanity check 1 fails (hidden state mismatch)**: Model architecture mismatch — verify the correct model is loaded for the `--model` preset.
- **Sanity check 4 fails (steering hook)**: The `MistralDecoderLayer` output format may have changed in a transformers update. Check that `output[0]` is the hidden states tensor.
- **Sanity check 5 fails (token round-trip)**: Tokenizer version mismatch. Ensure `transformers` version matches between vLLM and HF.
- **Full E2E hangs on vLLM init**: vLLM may be trying to use more GPU memory than available. Try `--gpu-memory-utilization 0.8` in the VLLMRunner or check if another process is using the GPU.
- **OOM during activation capture**: Reduce `--batch-size` (default 16). Or switch to the hook-based fallback documented in `activation_capture.py`.
