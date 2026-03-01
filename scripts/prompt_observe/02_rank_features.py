#!/usr/bin/env python3
"""Rank SAE features by property-specificity using differential analysis.

Three analysis modes:
  1. Per-property differential: rank features by target vs control signal
  2. Cross-property exclusivity: identify features unique to one property
  3. Aggregate: features that are strong across multiple properties

Runs on CPU using saved activation JSON files from 01_collect_activations.py.

Usage:
    python 02_rank_features.py
    python 02_rank_features.py --min-freq 0.3 --top-n 30
    python 02_rank_features.py --properties error_handling,recursion
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from glob import glob

sys.path.insert(0, os.path.dirname(__file__))
from properties import PROPERTIES


def load_activations(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_differential(
    target_results: list[dict],
    control_results: list[dict],
    feature_key: str,
    min_freq: float,
    top_n: int,
) -> list[dict]:
    """Compute differential score for each feature: target signal - control signal.

    Follows the same formula used in 05_prompt_and_observe.py and
    typing/02_find_typing_features.py for consistency.
    """
    n_target = len(target_results)
    n_control = len(control_results)

    # Accumulate per-feature stats from target prompts
    target_counts = Counter()
    target_activations = defaultdict(list)
    for result in target_results:
        for feat in result.get(feature_key, []):
            idx = feat["feature_idx"]
            target_counts[idx] += 1
            target_activations[idx].append(feat["activation"])

    # Accumulate per-feature stats from control prompts
    control_counts = Counter()
    control_activations = defaultdict(list)
    for result in control_results:
        for feat in result.get(feature_key, []):
            idx = feat["feature_idx"]
            control_counts[idx] += 1
            control_activations[idx].append(feat["activation"])

    all_features = set(target_counts.keys()) | set(control_counts.keys())

    ranked = []
    for feat_idx in all_features:
        target_freq = target_counts[feat_idx] / n_target if n_target > 0 else 0
        control_freq = control_counts[feat_idx] / n_control if n_control > 0 else 0

        target_mean = (
            sum(target_activations[feat_idx]) / len(target_activations[feat_idx])
            if target_activations[feat_idx] else 0
        )
        control_mean = (
            sum(control_activations[feat_idx]) / len(control_activations[feat_idx])
            if control_activations[feat_idx] else 0
        )

        target_signal = target_freq * target_mean
        control_signal = control_freq * control_mean
        diff_score = target_signal - control_signal

        if diff_score <= 0 or target_freq < min_freq:
            continue

        ranked.append({
            "feature_idx": int(feat_idx),
            "differential_score": round(diff_score, 4),
            "target_frequency": round(target_freq, 4),
            "target_mean_activation": round(target_mean, 4),
            "target_count": target_counts[feat_idx],
            "control_frequency": round(control_freq, 4),
            "control_mean_activation": round(control_mean, 4),
            "control_count": control_counts[feat_idx],
            "specificity_ratio": round(
                target_signal / (control_signal + 1e-8), 2
            ),
        })

    ranked.sort(key=lambda x: x["differential_score"], reverse=True)
    return ranked[:top_n]


def print_ranked_table(features: list[dict], title: str, max_rows: int = 20):
    """Print a formatted ranking table."""
    print(f"\n{'='*90}")
    print(title)
    print(f"{'='*90}")
    header = (
        f"{'Rank':<5} {'Feature':<10} {'DiffScore':<11} "
        f"{'TgtFreq':<9} {'TgtAct':<9} "
        f"{'CtlFreq':<9} {'CtlAct':<9} {'Ratio':<8}"
    )
    print(header)
    print("-" * 90)

    for i, feat in enumerate(features[:max_rows], 1):
        print(
            f"{i:<5} "
            f"{feat['feature_idx']:<10} "
            f"{feat['differential_score']:<11.4f} "
            f"{feat['target_frequency']:<9.4f} "
            f"{feat['target_mean_activation']:<9.4f} "
            f"{feat['control_frequency']:<9.4f} "
            f"{feat['control_mean_activation']:<9.4f} "
            f"{feat['specificity_ratio']:<8.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Rank features by property-specificity"
    )
    parser.add_argument("--top-n", type=int, default=50,
                        help="Number of top features per property")
    parser.add_argument("--min-freq", type=float, default=0.2,
                        help="Minimum target frequency threshold")
    parser.add_argument("--feature-key", type=str, default="mean_pooled_features",
                        choices=["mean_pooled_features", "last_token_features"],
                        help="Which activation type to analyze")
    parser.add_argument("--properties", type=str, default=None,
                        help="Comma-separated list of properties to analyze (default: all found)")
    parser.add_argument("--results-dir", type=str,
                        default="scripts/prompt_observe/results",
                        help="Results directory")
    args = parser.parse_args()

    # Discover available activation files
    activation_files = sorted(glob(
        os.path.join(args.results_dir, "activations_*.json")
    ))

    if not activation_files:
        print("Error: No activation files found. Run 01_collect_activations.py first.")
        return

    # Filter to requested properties
    available = {}
    for path in activation_files:
        filename = os.path.basename(path)
        prop_name = filename.replace("activations_", "").replace(".json", "")
        if prop_name in PROPERTIES:
            available[prop_name] = path

    if args.properties:
        requested = args.properties.split(",")
        available = {k: v for k, v in available.items() if k in requested}

    print(f"Found activation data for {len(available)} properties: {list(available.keys())}")
    print(f"Analysis config: feature_key={args.feature_key}, min_freq={args.min_freq}, top_n={args.top_n}")

    # ── Mode 1: Per-property differential ranking ──
    print(f"\n\n{'#'*90}")
    print("MODE 1: Per-Property Differential Ranking")
    print(f"{'#'*90}")

    per_property_results = {}

    for prop_name, path in available.items():
        data = load_activations(path)
        ranked = compute_differential(
            data["target_results"],
            data["control_results"],
            args.feature_key,
            args.min_freq,
            args.top_n,
        )
        per_property_results[prop_name] = ranked
        print_ranked_table(ranked, f"Property: {prop_name} ({data['description']})")

    # ── Mode 2: Cross-property exclusivity ──
    print(f"\n\n{'#'*90}")
    print("MODE 2: Cross-Property Analysis")
    print(f"{'#'*90}")

    # For each feature, track which properties it appears in
    feature_property_map = defaultdict(lambda: {
        "properties": [],
        "best_score": 0,
        "best_property": None,
        "total_score": 0,
    })

    for prop_name, ranked in per_property_results.items():
        top_set = {f["feature_idx"] for f in ranked[:args.top_n]}
        for feat in ranked:
            idx = feat["feature_idx"]
            entry = feature_property_map[idx]
            entry["properties"].append(prop_name)
            entry["total_score"] += feat["differential_score"]
            if feat["differential_score"] > entry["best_score"]:
                entry["best_score"] = feat["differential_score"]
                entry["best_property"] = prop_name

    # Exclusive features: appear in exactly one property's top results
    exclusive = []
    for feat_idx, info in feature_property_map.items():
        if len(info["properties"]) == 1:
            exclusive.append({
                "feature_idx": feat_idx,
                "property": info["properties"][0],
                "differential_score": info["best_score"],
            })
    exclusive.sort(key=lambda x: x["differential_score"], reverse=True)

    print(f"\nExclusive features (appear in only one property's top-{args.top_n}):")
    print(f"{'Feature':<10} {'Property':<25} {'DiffScore':<12}")
    print("-" * 47)
    for feat in exclusive[:30]:
        print(
            f"{feat['feature_idx']:<10} "
            f"{feat['property']:<25} "
            f"{feat['differential_score']:<12.4f}"
        )

    # Shared features: appear across multiple properties
    shared = []
    for feat_idx, info in feature_property_map.items():
        if len(info["properties"]) > 1:
            shared.append({
                "feature_idx": feat_idx,
                "n_properties": len(info["properties"]),
                "properties": sorted(info["properties"]),
                "best_property": info["best_property"],
                "best_score": info["best_score"],
                "total_score": round(info["total_score"], 4),
            })
    shared.sort(key=lambda x: x["n_properties"], reverse=True)

    print(f"\nShared features (appear across multiple properties):")
    print(f"{'Feature':<10} {'#Props':<8} {'BestProp':<25} {'TotalScore':<12} {'Properties'}")
    print("-" * 90)
    for feat in shared[:20]:
        print(
            f"{feat['feature_idx']:<10} "
            f"{feat['n_properties']:<8} "
            f"{feat['best_property']:<25} "
            f"{feat['total_score']:<12.4f} "
            f"{', '.join(feat['properties'])}"
        )

    # ── Mode 3: Aggregate ranking ──
    print(f"\n\n{'#'*90}")
    print("MODE 3: Aggregate Ranking (features sorted by total differential across properties)")
    print(f"{'#'*90}")

    aggregate = []
    for feat_idx, info in feature_property_map.items():
        aggregate.append({
            "feature_idx": feat_idx,
            "total_differential": round(info["total_score"], 4),
            "n_properties": len(info["properties"]),
            "best_property": info["best_property"],
            "best_score": round(info["best_score"], 4),
            "properties": sorted(info["properties"]),
        })
    aggregate.sort(key=lambda x: x["total_differential"], reverse=True)

    print(f"\n{'Rank':<5} {'Feature':<10} {'TotalDiff':<12} {'#Props':<8} {'BestProp':<20} {'BestScore':<12}")
    print("-" * 67)
    for i, feat in enumerate(aggregate[:30], 1):
        print(
            f"{i:<5} "
            f"{feat['feature_idx']:<10} "
            f"{feat['total_differential']:<12.4f} "
            f"{feat['n_properties']:<8} "
            f"{feat['best_property']:<20} "
            f"{feat['best_score']:<12.4f}"
        )

    # ── Save results ──
    output = {
        "analysis_config": {
            "feature_key": args.feature_key,
            "min_freq": args.min_freq,
            "top_n": args.top_n,
            "properties_analyzed": list(available.keys()),
        },
        "per_property": per_property_results,
        "cross_property": {
            "exclusive_features": exclusive,
            "shared_features": shared,
        },
        "aggregate_ranking": aggregate[:args.top_n],
    }

    output_path = os.path.join(args.results_dir, "features_ranked.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved full analysis to {output_path}")

    # Print final summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for prop_name in available:
        n_feats = len(per_property_results.get(prop_name, []))
        top3 = per_property_results.get(prop_name, [])[:3]
        top3_str = ", ".join(
            f"{f['feature_idx']}({f['differential_score']:.3f})"
            for f in top3
        )
        print(f"  {prop_name}: {n_feats} features, top = [{top3_str}]")
    print(f"\n  Exclusive features: {len(exclusive)}")
    print(f"  Shared features: {len(shared)}")
    print(f"  Total unique features: {len(feature_property_map)}")


if __name__ == "__main__":
    main()
