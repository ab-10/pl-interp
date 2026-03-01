"""Difference-in-means contrastive directions. For each prompt variant, computes
mean(variant activations) - mean(baseline activations) to obtain a steering
direction in residual stream space."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments import config
from experiments.storage.activation_store import ActivationReader
from experiments.storage.schema import GenerationRecord, read_records

logger = logging.getLogger(__name__)

HIDDEN_DIM = config.MODEL_HIDDEN_DIM  # 4096


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _iter_generation_files(generations_dir: Path) -> list[Path]:
    """Collect all JSONL generation files."""
    return sorted(generations_dir.glob("*.jsonl"))


def _load_all_records(generations_dir: Path) -> list[GenerationRecord]:
    """Load all generation records from all JSONL files in directory."""
    records: list[GenerationRecord] = []
    for path in _iter_generation_files(generations_dir):
        records.extend(read_records(path))
    return records


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_contrastive_directions(
    generations_dir: Path,
    activations_dir: Path,
    output_dir: Path,
    device: str = "cpu",
) -> Path:
    """Compute difference-in-means steering directions for each variant.

    For each non-baseline variant, the direction is:
        direction = mean(variant activations) - mean(baseline activations)
    normalized to unit norm.

    Args:
        generations_dir: Directory containing generation record JSONL files.
        activations_dir: Directory containing activation shard files.
        output_dir: Directory for output files.
        device: Torch device for final tensors (default: cpu).

    Returns:
        Path to the saved contrastive_directions.pt file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading generation records from %s", generations_dir)
    records = _load_all_records(generations_dir)
    logger.info("Loaded %d records", len(records))

    # Running sums (float64 for numerical stability) and token counts per variant
    sum_vectors: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(HIDDEN_DIM, dtype=np.float64)
    )
    token_counts: dict[str, int] = defaultdict(int)

    # Cache activation readers per shard file
    readers: dict[str, ActivationReader] = {}

    for rec_idx, record in enumerate(records):
        if rec_idx % 1000 == 0:
            logger.info("Processing record %d / %d", rec_idx, len(records))

        # Resolve activation shard path
        act_file = record.activation_file
        if act_file not in readers:
            act_path = Path(act_file)
            if not act_path.is_absolute():
                act_path = activations_dir / act_path
            readers[act_file] = ActivationReader(act_path)

        reader = readers[act_file]
        act_np = reader.read(record.activation_offset, record.activation_length)

        num_tokens = act_np.shape[0]
        if num_tokens == 0:
            continue

        # Sum across token positions (cast to float64 for stability), add to variant
        token_sum = act_np.astype(np.float64).sum(axis=0)  # shape (4096,)
        sum_vectors[record.variant_id] += token_sum
        token_counts[record.variant_id] += num_tokens

    # Compute per-variant means
    mean_vectors: dict[str, np.ndarray] = {}
    for variant_id in sum_vectors:
        count = token_counts[variant_id]
        if count > 0:
            mean_vectors[variant_id] = sum_vectors[variant_id] / count
        else:
            mean_vectors[variant_id] = np.zeros(HIDDEN_DIM, dtype=np.float64)
        logger.info(
            "Variant %s: %d tokens, mean norm %.4f",
            variant_id,
            count,
            np.linalg.norm(mean_vectors[variant_id]),
        )

    if "baseline" not in mean_vectors:
        raise ValueError("No baseline records found in generation data")

    mean_baseline = mean_vectors["baseline"]

    # Compute directions for non-baseline variants
    directions: dict[str, torch.Tensor] = {}
    norms: dict[str, float] = {}

    non_baseline_variants = [v for v in config.VARIANT_IDS if v != "baseline"]

    for variant_id in non_baseline_variants:
        if variant_id not in mean_vectors:
            logger.warning("Variant %s has no records, skipping", variant_id)
            continue

        direction = mean_vectors[variant_id] - mean_baseline
        pre_norm = float(np.linalg.norm(direction))
        norms[variant_id] = pre_norm

        if pre_norm < 1e-12:
            logger.warning("Variant %s has near-zero direction norm %.2e", variant_id, pre_norm)
            direction_tensor = torch.zeros(HIDDEN_DIM, device=device)
        else:
            direction = direction / pre_norm
            direction_tensor = torch.tensor(direction, dtype=torch.float32, device=device)

        directions[variant_id] = direction_tensor
        logger.info("Variant %s: pre-norm %.4f", variant_id, pre_norm)

    # Save
    save_dict: dict = {**directions, "norms": norms}
    out_path = output_dir / "contrastive_directions.pt"
    torch.save(save_dict, out_path)
    logger.info("Saved %d directions to %s", len(directions), out_path)

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute difference-in-means contrastive steering directions.",
    )
    parser.add_argument(
        "--generations-dir",
        type=Path,
        required=True,
        help="Directory containing generation record JSONL files.",
    )
    parser.add_argument(
        "--activations-dir",
        type=Path,
        required=True,
        help="Directory containing activation shard files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    compute_contrastive_directions(
        generations_dir=args.generations_dir,
        activations_dir=args.activations_dir,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
