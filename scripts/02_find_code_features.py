#!/usr/bin/env python3
"""Find code-specific SAE features via differential analysis.

Loads activation data from script 01, computes frequency and mean activation
for each feature on code vs non-code prompts, and ranks features by a
differential score. Outputs the top 50 code-specific features.

Usage:
    python 02_find_code_features.py [--top-n 50] [--min-freq 0.05]
"""

import argparse
import json
import os
from collections import defaultdict


def load_activations(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def compute_feature_stats(activations: list[dict]) -> dict:
    """Compute per-feature frequency and mean activation.

    Returns dict mapping feature_idx -> {count, total_activation, prompts}
    """
    stats = defaultdict(lambda: {"count": 0, "total_activation": 0.0, "prompts": []})
    total_prompts = len(activations)

    for record in activations:
        for feat in record.get("top_features", []):
            idx = feat["feature_idx"]
            act = feat["activation"]
            stats[idx]["count"] += 1
            stats[idx]["total_activation"] += act
            stats[idx]["prompts"].append(record["id"])

    # Compute frequency and mean activation
    for idx in stats:
        stats[idx]["frequency"] = stats[idx]["count"] / total_prompts if total_prompts > 0 else 0
        stats[idx]["mean_activation"] = (
            stats[idx]["total_activation"] / stats[idx]["count"]
            if stats[idx]["count"] > 0
            else 0
        )

    return dict(stats)


def rank_features(
    code_stats: dict,
    noncode_stats: dict,
    n_code: int,
    n_noncode: int,
    min_code_freq: float,
    top_n: int,
) -> list[dict]:
    """Rank features by differential score: code_freq * code_mean - noncode_freq * noncode_mean."""
    all_features = set(code_stats.keys()) | set(noncode_stats.keys())
    ranked = []

    for idx in all_features:
        code = code_stats.get(idx, {"frequency": 0, "mean_activation": 0, "count": 0})
        noncode = noncode_stats.get(idx, {"frequency": 0, "mean_activation": 0, "count": 0})

        code_freq = code["frequency"]
        code_mean = code["mean_activation"]
        noncode_freq = noncode["frequency"]
        noncode_mean = noncode["mean_activation"]

        # Skip features that don't fire enough on code
        if code_freq < min_code_freq:
            continue

        # Differential score
        differential = code_freq * code_mean - noncode_freq * noncode_mean

        # Only keep features with positive differential (more code than non-code)
        if differential <= 0:
            continue

        ranked.append({
            "feature_idx": int(idx),
            "differential_score": round(differential, 4),
            "code_frequency": round(code_freq, 4),
            "code_mean_activation": round(code_mean, 4),
            "code_count": code["count"],
            "noncode_frequency": round(noncode_freq, 4),
            "noncode_mean_activation": round(noncode_mean, 4),
            "noncode_count": noncode["count"],
            "specificity_ratio": round(
                (code_freq * code_mean) / (noncode_freq * noncode_mean + 1e-8), 2
            ),
        })

    # Sort by differential score descending
    ranked.sort(key=lambda x: x["differential_score"], reverse=True)
    return ranked[:top_n]


def print_summary_table(features: list[dict], n_code: int, n_noncode: int):
    """Print a formatted summary table of top features."""
    print(f"\n{'='*90}")
    print(f"Top {len(features)} Code-Specific Features (from {n_code} code / {n_noncode} non-code prompts)")
    print(f"{'='*90}")
    print(
        f"{'Rank':<5} {'Feature':<10} {'Score':<10} "
        f"{'Code Freq':<10} {'Code Act':<10} "
        f"{'NC Freq':<10} {'NC Act':<10} {'Ratio':<8}"
    )
    print("-" * 90)

    for i, feat in enumerate(features, 1):
        print(
            f"{i:<5} {feat['feature_idx']:<10} {feat['differential_score']:<10.4f} "
            f"{feat['code_frequency']:<10.4f} {feat['code_mean_activation']:<10.4f} "
            f"{feat['noncode_frequency']:<10.4f} {feat['noncode_mean_activation']:<10.4f} "
            f"{feat['specificity_ratio']:<8.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Find code-specific SAE features")
    parser.add_argument("--top-n", type=int, default=50, help="Number of top features to output")
    parser.add_argument("--min-freq", type=float, default=0.05, help="Minimum code frequency threshold")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory")
    args = parser.parse_args()

    code_path = os.path.join(args.results_dir, "activations_code.json")
    noncode_path = os.path.join(args.results_dir, "activations_noncode.json")

    if not os.path.exists(code_path) or not os.path.exists(noncode_path):
        print("Error: activation files not found. Run 01_collect_activations.py first.")
        return

    # Load activation data
    print("Loading activation data...")
    code_data = load_activations(code_path)
    noncode_data = load_activations(noncode_path)

    n_code = len(code_data)
    n_noncode = len(noncode_data)
    print(f"  Code prompts: {n_code}")
    print(f"  Non-code prompts: {n_noncode}")

    # Filter out error entries
    code_data = [r for r in code_data if r.get("top_features")]
    noncode_data = [r for r in noncode_data if r.get("top_features")]
    print(f"  Code prompts with features: {len(code_data)}")
    print(f"  Non-code prompts with features: {len(noncode_data)}")

    # Compute per-feature stats
    print("\nComputing feature statistics...")
    code_stats = compute_feature_stats(code_data)
    noncode_stats = compute_feature_stats(noncode_data)

    print(f"  Unique features in code: {len(code_stats)}")
    print(f"  Unique features in non-code: {len(noncode_stats)}")

    # Rank features
    print(f"\nRanking features (min code freq: {args.min_freq})...")
    ranked = rank_features(
        code_stats, noncode_stats, n_code, n_noncode, args.min_freq, args.top_n
    )

    if not ranked:
        print("No features passed the filtering criteria!")
        return

    # Print summary
    print_summary_table(ranked, n_code, n_noncode)

    # Save results
    output_path = os.path.join(args.results_dir, "code_features_ranked.json")
    with open(output_path, "w") as f:
        json.dump(ranked, f, indent=2)
    print(f"\nSaved {len(ranked)} ranked features to {output_path}")


if __name__ == "__main__":
    main()
