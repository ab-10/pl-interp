"""Select steering feature candidates from SAE feature statistics. Picks 3 diverse
high-effect-size features for steering and 3 random control features with no pass/fail
signal, then exports their decoder directions as a torch file."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import torch

from experiments.sae.model import TopKSAE

logger = logging.getLogger(__name__)

# Threshold below which a feature is considered dead (sum of mean_pass + mean_fail)
_DEAD_THRESHOLD = 1e-8

# Cohen's d threshold for "no signal" control features
_CONTROL_D_THRESHOLD = 0.1


def _load_sae(checkpoint_path: Path, device: str) -> TopKSAE:
    """Load a trained TopKSAE from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sae = TopKSAE(d_model=cfg["d_model"], d_sae=cfg["d_sae"], k=cfg["k"])
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(device)
    sae.eval()
    return sae


def _primary_variant(feature: dict) -> str:
    """Return the variant with the highest mean activation for a feature."""
    v_means = feature.get("variant_means", {})
    if not v_means:
        return ""
    return max(v_means, key=lambda v: v_means[v])


def select_steering_candidates(
    feature_stats_path: Path,
    sae_checkpoint: Path,
    output_dir: Path,
    device: str = "cpu",
    seed: int = 42,
) -> Path:
    """Select 3 steering candidates + 3 random controls from feature statistics.

    Selection strategy:
      1. Filter dead features (mean_pass + mean_fail near zero).
      2. Rank by |Cohen's d|.
      3. From top 50, pick 3 candidates with variant diversity — each candidate's
         primary variant (variant with highest mean activation) should differ.
         Falls back to top-3 by |Cohen's d| if diversity is insufficient.
      4. Pick 3 random control features from the middle of the distribution
         (|Cohen's d| < 0.1) for matched-norm random direction baselines.

    Args:
        feature_stats_path: Path to feature_stats.json from analyze_features.
        sae_checkpoint: Path to trained SAE checkpoint (.pt).
        output_dir: Directory for output files.
        device: Torch device for loading the SAE.
        seed: Random seed for control feature selection.

    Returns:
        Path to the written feature_candidates.json file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    logger.info("Loading feature statistics from %s", feature_stats_path)
    with open(feature_stats_path) as f:
        stats = json.load(f)

    features: list[dict] = stats["features"]

    # Filter dead features
    alive = [
        feat for feat in features
        if abs(feat["mean_pass"]) + abs(feat["mean_fail"]) > _DEAD_THRESHOLD
    ]
    logger.info("Alive features: %d / %d", len(alive), len(features))

    # Rank by |Cohen's d| descending
    alive.sort(key=lambda feat: abs(feat["cohens_d"]), reverse=True)

    # --- Select 3 candidates with variant diversity from top 50 ---
    top_50 = alive[:50]

    # Attempt diversity selection
    candidates: list[dict] = []
    used_variants: set[str] = set()

    for feat in top_50:
        pv = _primary_variant(feat)
        if pv not in used_variants:
            candidates.append(feat)
            used_variants.add(pv)
        if len(candidates) == 3:
            break

    # If diversity wasn't enough, fill remaining slots with top features by |d|
    if len(candidates) < 3:
        selected_idxs = {c["feature_idx"] for c in candidates}
        for feat in top_50:
            if feat["feature_idx"] not in selected_idxs:
                candidates.append(feat)
                selected_idxs.add(feat["feature_idx"])
            if len(candidates) == 3:
                break

    logger.info(
        "Selected candidates: %s",
        [(c["feature_idx"], round(c["cohens_d"], 4), _primary_variant(c)) for c in candidates],
    )

    # --- Select 3 random control features with |d| < threshold ---
    control_pool = [
        feat for feat in alive
        if abs(feat["cohens_d"]) < _CONTROL_D_THRESHOLD
    ]
    candidate_idxs = {c["feature_idx"] for c in candidates}
    control_pool = [feat for feat in control_pool if feat["feature_idx"] not in candidate_idxs]

    if len(control_pool) >= 3:
        control_features = rng.sample(control_pool, 3)
    else:
        logger.warning(
            "Only %d features with |d| < %.2f; using all available for controls",
            len(control_pool),
            _CONTROL_D_THRESHOLD,
        )
        control_features = control_pool

    random_control_idxs = [feat["feature_idx"] for feat in control_features]
    logger.info("Random control features: %s", random_control_idxs)

    # --- Extract steering directions from SAE decoder ---
    logger.info("Loading SAE for direction extraction from %s", sae_checkpoint)
    sae = _load_sae(sae_checkpoint, device)
    W_dec = sae.W_dec.detach()  # shape (d_sae, d_model)

    directions: dict[int, torch.Tensor] = {}
    for c in candidates:
        idx = c["feature_idx"]
        directions[idx] = W_dec[idx].clone().cpu()

    random_directions: dict[int, torch.Tensor] = {}
    for idx in random_control_idxs:
        random_directions[idx] = W_dec[idx].clone().cpu()

    # --- Write outputs ---
    candidates_output = {
        "candidates": [
            {
                "feature_idx": c["feature_idx"],
                "cohens_d": c["cohens_d"],
                "primary_variant": _primary_variant(c),
                "variant_means": c["variant_means"],
                "top_examples": c["top_examples"],
            }
            for c in candidates
        ],
        "random_control_features": random_control_idxs,
    }

    candidates_path = output_dir / "feature_candidates.json"
    with open(candidates_path, "w") as f:
        json.dump(candidates_output, f, indent=2)
    logger.info("Wrote %s", candidates_path)

    directions_path = output_dir / "steering_directions.pt"
    torch.save(
        {"directions": directions, "random_directions": random_directions},
        directions_path,
    )
    logger.info("Wrote %s", directions_path)

    return candidates_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select 3 steering feature candidates + 3 random controls from feature stats.",
    )
    parser.add_argument(
        "--feature-stats",
        type=Path,
        required=True,
        help="Path to feature_stats.json from analyze_features.",
    )
    parser.add_argument(
        "--sae-checkpoint",
        type=Path,
        required=True,
        help="Path to trained SAE checkpoint (.pt).",
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
        help="Torch device for loading SAE (default: cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for control feature selection (default: 42).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    select_steering_candidates(
        feature_stats_path=args.feature_stats,
        sae_checkpoint=args.sae_checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
