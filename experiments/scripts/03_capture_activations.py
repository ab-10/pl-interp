"""Stage 3: Capture multi-layer activations via HF teacher-forcing.

Reads: /scratch/<model>/generations/shard_{N}.jsonl (evaluated records with gen_token_ids)
Writes: /scratch/<model>/activations/layer_{L}/shard_{N}.npy per capture layer
Updates: shard JSONL with activation_layers metadata

Uses HF model (not vLLM) for forward pass with output_hidden_states=True.
Sharded by GPU — each shard processes its own generation records.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m experiments.scripts.03_capture_activations --model ministral-8b --shard 0
  CUDA_VISIBLE_DEVICES=1 python -m experiments.scripts.03_capture_activations --model ministral-8b --shard 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import wandb

from experiments import config
from experiments.generation.activation_capture import ActivationCapture
from experiments.storage.activation_store import ActivationWriter
from experiments.storage.schema import read_records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 3: Capture multi-layer activations via HF teacher-forcing.",
    )
    config.add_model_arg(parser)
    parser.add_argument(
        "--shard", type=int, required=True,
        help="GPU shard index (0-based), matches generation shard",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Records per forward pass (default: 16)",
    )
    args = parser.parse_args()

    config.apply_args(args)

    # --- W&B ---
    run = wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        name=f"03_capture_{config.MODEL_NAME}_shard{args.shard}",
        config={
            "stage": "03_capture_activations",
            "model": config.MODEL_NAME,
            "model_id": config.MODEL_ID,
            "shard": args.shard,
            "capture_layers": config.CAPTURE_LAYERS,
            "hidden_states_indices": config.HIDDEN_STATES_INDICES,
            "hidden_dim": config.MODEL_HIDDEN_DIM,
            "batch_size": args.batch_size,
        },
    )

    # --- Load evaluated records ---
    print(f"Model: {config.MODEL_NAME} (capture layers {config.CAPTURE_LAYERS})")
    shard_file = config.GENERATIONS_DIR / f"shard_{args.shard}.jsonl"
    if not shard_file.exists():
        print(f"Shard file not found: {shard_file}")
        return 1

    records = read_records(shard_file)
    print(f"Loaded {len(records)} records from {shard_file}")

    # Skip records that already have activations (for resume)
    pending = [r for r in records if not r.activation_layers]
    if len(pending) < len(records):
        print(f"  Skipping {len(records) - len(pending)} already-captured records")
    if not pending:
        print("All records already have activations. Nothing to do.")
        return 0

    # --- Initialize model ---
    print(f"Loading model {config.MODEL_ID} for activation capture...")
    capture = ActivationCapture()

    # --- Initialize per-layer activation writers ---
    writers: dict[int, ActivationWriter] = {}
    act_paths: dict[int, Path] = {}
    for layer_num in config.CAPTURE_LAYERS:
        layer_dir = config.ACTIVATIONS_DIR / f"layer_{layer_num}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        act_path = layer_dir / f"shard_{args.shard}.npy"
        writers[layer_num] = ActivationWriter(act_path)
        act_paths[layer_num] = act_path
        print(f"  Layer {layer_num} -> {act_path}")

    # --- Capture in batches ---
    t0 = time.time()
    total_tokens = 0

    for i in range(0, len(pending), args.batch_size):
        batch = pending[i : i + args.batch_size]

        # Returns list of {layer_num: np.ndarray} per record
        activations = capture.capture_batch(batch, batch_size=len(batch))

        for record, layer_acts in zip(batch, activations):
            record.activation_layers = {}
            for layer_num, acts in layer_acts.items():
                offset, length = writers[layer_num].append(acts)
                record.activation_layers[layer_num] = {
                    "file": str(act_paths[layer_num]),
                    "offset": offset,
                    "length": length,
                }
            # Token count is same across layers; use first
            first_layer = config.CAPTURE_LAYERS[0]
            total_tokens += record.activation_layers[first_layer]["length"]

        done = i + len(batch)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  {done}/{len(pending)} records ({total_tokens} tokens, {rate:.1f} rec/s)")

    elapsed = time.time() - t0
    layers_str = "+".join(str(l) for l in config.CAPTURE_LAYERS)
    print(f"\nCapture complete: {len(pending)} records, {total_tokens} tokens, "
          f"layers [{layers_str}] in {elapsed:.1f}s")

    # --- Write updated records (atomic: tmp + rename) ---
    tmp_path = shard_file.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        for record in records:
            f.write(record.to_json_line() + "\n")
    tmp_path.rename(shard_file)
    print(f"Updated records saved to {shard_file}")

    # --- W&B summary ---
    wandb.log({
        "total_records": len(pending),
        "total_tokens": total_tokens,
        "capture_time_s": elapsed,
    })
    wandb.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
