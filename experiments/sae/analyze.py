"""Feature-level statistics for SAE latents. Streams through generation records and
computes per-feature Cohen's d (pass vs fail), variant correlations, and top activating
examples using online algorithms for memory efficiency."""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from experiments.sae.model import TopKSAE
from experiments.storage.activation_store import ActivationReader
from experiments.storage.schema import GenerationRecord, read_records

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Online statistics (Welford's algorithm)
# ---------------------------------------------------------------------------

@dataclass
class WelfordAccumulator:
    """Running mean and variance via Welford's online algorithm."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / self.count

    def std(self) -> float:
        return math.sqrt(self.variance())


def _cohens_d(acc_pass: WelfordAccumulator, acc_fail: WelfordAccumulator) -> float:
    """Cohen's d = (mean_pass - mean_fail) / pooled_std."""
    n_p, n_f = acc_pass.count, acc_fail.count
    if n_p < 2 or n_f < 2:
        return 0.0
    # Use m2 directly: m2 = sum of squared deviations from mean.
    # Pooled variance = (m2_pass + m2_fail) / (n_pass + n_fail - 2)
    pooled_var = (acc_pass.m2 + acc_fail.m2) / (n_p + n_f - 2)
    pooled_std = math.sqrt(pooled_var)
    if pooled_std < 1e-12:
        return 0.0
    return (acc_pass.mean - acc_fail.mean) / pooled_std


# ---------------------------------------------------------------------------
# Top-K example tracking (min-heap of size 10)
# ---------------------------------------------------------------------------

@dataclass(order=True)
class TopExample:
    """Sortable container for min-heap (sorted by value ascending so min is evicted)."""

    value: float
    task_id: str = field(compare=False)
    variant_id: str = field(compare=False)
    run_id: int = field(compare=False)
    position: int = field(compare=False)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "variant_id": self.variant_id,
            "run_id": self.run_id,
            "position": self.position,
            "value": round(self.value, 4),
        }


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_sae(checkpoint_path: Path, device: str) -> TopKSAE:
    """Load a trained TopKSAE from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sae = TopKSAE(d_model=cfg["d_model"], d_sae=cfg["d_sae"], k=cfg["k"])
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(device)
    sae.eval()
    return sae


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
# Main analysis
# ---------------------------------------------------------------------------

def analyze_features(
    sae_checkpoint: Path,
    generations_dir: Path,
    activations_dir: Path,
    output_dir: Path,
    batch_size: int = 512,
    device: str = "cuda",
) -> Path:
    """Compute per-feature statistics by streaming through all generation records.

    For each of d_sae features, computes:
      - Cohen's d (pass vs fail effect size)
      - Mean activation per variant
      - Top 10 activating examples

    Args:
        sae_checkpoint: Path to the trained SAE checkpoint (.pt file).
        generations_dir: Directory containing generation record JSONL files.
        activations_dir: Directory containing activation shard files.
        output_dir: Directory for output files.
        batch_size: Number of activation rows to process at once through the SAE.
        device: Torch device ("cuda" or "cpu").

    Returns:
        Path to the written feature_stats.json file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SAE from %s", sae_checkpoint)
    sae = _load_sae(sae_checkpoint, device)
    d_sae = sae.W_dec.shape[0]

    logger.info("Loading generation records from %s", generations_dir)
    records = _load_all_records(generations_dir)
    logger.info("Loaded %d records", len(records))

    # Per-feature accumulators
    acc_pass: list[WelfordAccumulator] = [WelfordAccumulator() for _ in range(d_sae)]
    acc_fail: list[WelfordAccumulator] = [WelfordAccumulator() for _ in range(d_sae)]

    # Per-feature per-variant running sums (divided by total tokens per variant at the end)
    variant_sums: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(d_sae, dtype=np.float64))

    # Per-feature top-10 min-heaps
    top_examples: list[list[TopExample]] = [[] for _ in range(d_sae)]
    top_k = 10

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

        # Process in sub-batches through SAE
        num_tokens = act_np.shape[0]
        if num_tokens == 0:
            continue

        for start in range(0, num_tokens, batch_size):
            end = min(start + batch_size, num_tokens)
            chunk_np = act_np[start:end]
            chunk_tensor = torch.from_numpy(chunk_np.astype(np.float32)).to(device)

            with torch.no_grad():
                _x_hat, _topk_latents, info = sae(chunk_tensor)

            # Only process the K nonzero features per token for efficiency
            chunk_size = info["topk_indices"].shape[0]
            for token_offset in range(chunk_size):
                indices = info["topk_indices"][token_offset].cpu().numpy()
                values = info["topk_values"][token_offset].cpu().numpy()

                for feat_idx, feat_val in zip(indices, values):
                    feat_idx = int(feat_idx)
                    feat_val = float(feat_val)

                    # Pass/fail accumulators
                    if record.passed:
                        acc_pass[feat_idx].update(feat_val)
                    else:
                        acc_fail[feat_idx].update(feat_val)

                    # Variant accumulator (sum; divided by total tokens at the end)
                    variant_sums[record.variant_id][feat_idx] += feat_val

                    # Top examples (min-heap)
                    example = TopExample(
                        value=feat_val,
                        task_id=record.task_id,
                        variant_id=record.variant_id,
                        run_id=record.run_id,
                        position=start + token_offset,
                    )
                    heap = top_examples[feat_idx]
                    if len(heap) < top_k:
                        heapq.heappush(heap, example)
                    elif feat_val > heap[0].value:
                        heapq.heapreplace(heap, example)

    # --- Build output ---
    logger.info("Computing final statistics for %d features", d_sae)

    # Variant IDs that appeared
    all_variant_ids = sorted(variant_sums.keys())

    # Total tokens per variant (denominator for variant mean activations)
    variant_total_tokens: dict[str, int] = defaultdict(int)
    for record in records:
        variant_total_tokens[record.variant_id] += record.activation_length

    features_list: list[dict] = []
    dead_count = 0
    positive_d_count = 0

    for f in range(d_sae):
        d = _cohens_d(acc_pass[f], acc_fail[f])
        mean_pass = acc_pass[f].mean if acc_pass[f].count > 0 else 0.0
        mean_fail = acc_fail[f].mean if acc_fail[f].count > 0 else 0.0

        # Check dead feature
        is_dead = (acc_pass[f].count == 0 and acc_fail[f].count == 0)
        if is_dead:
            dead_count += 1

        if d > 0:
            positive_d_count += 1

        # Variant means
        v_means: dict[str, float] = {}
        for v in all_variant_ids:
            total_tok = variant_total_tokens.get(v, 0)
            if total_tok > 0:
                v_means[v] = round(float(variant_sums[v][f]) / total_tok, 6)
            else:
                v_means[v] = 0.0

        # Top examples (sorted descending by value)
        sorted_examples = sorted(top_examples[f], key=lambda e: e.value, reverse=True)

        features_list.append({
            "feature_idx": f,
            "cohens_d": round(d, 6),
            "mean_pass": round(mean_pass, 6),
            "mean_fail": round(mean_fail, 6),
            "variant_means": v_means,
            "top_examples": [ex.to_dict() for ex in sorted_examples],
        })

    output = {
        "features": features_list,
        "summary": {
            "total_features": d_sae,
            "dead_features": dead_count,
            "features_with_positive_d": positive_d_count,
        },
    }

    out_path = output_dir / "feature_stats.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(
        "Wrote %s — %d features, %d dead, %d with positive Cohen's d",
        out_path,
        d_sae,
        dead_count,
        positive_d_count,
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-feature SAE statistics (Cohen's d, variant means, top examples).",
    )
    parser.add_argument(
        "--sae-checkpoint",
        type=Path,
        required=True,
        help="Path to trained SAE checkpoint (.pt).",
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
        "--batch-size",
        type=int,
        default=512,
        help="Number of activation rows per SAE forward pass (default: 512).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (default: cuda).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    analyze_features(
        sae_checkpoint=args.sae_checkpoint,
        generations_dir=args.generations_dir,
        activations_dir=args.activations_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
