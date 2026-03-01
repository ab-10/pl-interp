#!/usr/bin/env python3
"""Find typing-specific SAE features via differential and paired analysis.

Three analysis modes:
  1. Combined ranking: pool all typed vs all untyped
  2. Per-language-family ranking: separate TS/JS and Python analyses
  3. Paired difference analysis: paired t-test per feature across matched pairs

Features appearing in the top results of ALL modes are consensus features.

Requires scipy for paired t-test.

Usage:
    python 02_find_typing_features.py [--top-n 50] [--min-freq 0.03]
"""

import argparse
import json
import os
from collections import defaultdict

from scipy import stats


def load_activations(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def compute_feature_stats(activations: list[dict], feature_key: str = "top_features") -> dict:
    """Compute per-feature frequency and mean activation."""
    feat_stats = defaultdict(lambda: {"count": 0, "total_activation": 0.0, "prompts": []})
    total_prompts = len(activations)

    for record in activations:
        for feat in record.get(feature_key, []):
            idx = feat["feature_idx"]
            act = feat["activation"]
            feat_stats[idx]["count"] += 1
            feat_stats[idx]["total_activation"] += act
            feat_stats[idx]["prompts"].append(record["pair_id"])

    for idx in feat_stats:
        feat_stats[idx]["frequency"] = feat_stats[idx]["count"] / total_prompts if total_prompts > 0 else 0
        feat_stats[idx]["mean_activation"] = (
            feat_stats[idx]["total_activation"] / feat_stats[idx]["count"]
            if feat_stats[idx]["count"] > 0 else 0
        )

    return dict(feat_stats)


def rank_features(
    typed_stats: dict,
    untyped_stats: dict,
    min_freq: float,
    top_n: int,
) -> list[dict]:
    """Rank features by differential score: typed_freq * typed_mean - untyped_freq * untyped_mean."""
    all_features = set(typed_stats.keys()) | set(untyped_stats.keys())
    ranked = []

    for idx in all_features:
        typed = typed_stats.get(idx, {"frequency": 0, "mean_activation": 0, "count": 0})
        untyped = untyped_stats.get(idx, {"frequency": 0, "mean_activation": 0, "count": 0})

        typed_freq = typed["frequency"]
        typed_mean = typed["mean_activation"]
        untyped_freq = untyped["frequency"]
        untyped_mean = untyped["mean_activation"]

        if typed_freq < min_freq:
            continue

        differential = typed_freq * typed_mean - untyped_freq * untyped_mean
        if differential <= 0:
            continue

        ranked.append({
            "feature_idx": int(idx),
            "differential_score": round(differential, 4),
            "typed_frequency": round(typed_freq, 4),
            "typed_mean_activation": round(typed_mean, 4),
            "typed_count": typed["count"],
            "untyped_frequency": round(untyped_freq, 4),
            "untyped_mean_activation": round(untyped_mean, 4),
            "untyped_count": untyped["count"],
            "specificity_ratio": round(
                (typed_freq * typed_mean) / (untyped_freq * untyped_mean + 1e-8), 2
            ),
        })

    ranked.sort(key=lambda x: x["differential_score"], reverse=True)
    return ranked[:top_n]


def paired_analysis(
    typed_data: list[dict],
    untyped_data: list[dict],
    feature_key: str = "top_features",
    top_n: int = 50,
) -> list[dict]:
    """Paired t-test analysis across matched typed/untyped pairs.

    For each feature, computes the activation difference within each pair,
    then runs a one-sample t-test on those differences.
    """
    # Build lookup: pair_id -> {feature_idx: activation}
    typed_by_pair = {}
    for record in typed_data:
        feat_map = {}
        for feat in record.get(feature_key, []):
            feat_map[feat["feature_idx"]] = feat["activation"]
        typed_by_pair[record["pair_id"]] = feat_map

    untyped_by_pair = {}
    for record in untyped_data:
        feat_map = {}
        for feat in record.get(feature_key, []):
            feat_map[feat["feature_idx"]] = feat["activation"]
        untyped_by_pair[record["pair_id"]] = feat_map

    # Get all pair IDs present in both sets
    common_pairs = sorted(set(typed_by_pair.keys()) & set(untyped_by_pair.keys()))
    if not common_pairs:
        print("Warning: No common pair IDs found between typed and untyped data!")
        return []

    # Collect all feature indices seen in either set
    all_features = set()
    for feat_map in typed_by_pair.values():
        all_features.update(feat_map.keys())
    for feat_map in untyped_by_pair.values():
        all_features.update(feat_map.keys())

    # Compute paired differences for each feature
    results = []
    for feat_idx in all_features:
        diffs = []
        for pair_id in common_pairs:
            typed_act = typed_by_pair[pair_id].get(feat_idx, 0.0)
            untyped_act = untyped_by_pair[pair_id].get(feat_idx, 0.0)
            diffs.append(typed_act - untyped_act)

        mean_diff = sum(diffs) / len(diffs)

        # Only analyze features with positive mean difference (typing-specific)
        if mean_diff <= 0:
            continue

        # Paired one-sample t-test (H0: mean difference = 0)
        t_stat, p_value = stats.ttest_1samp(diffs, 0.0)

        # Cohen's d effect size
        std_diff = (sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

        # Count pairs where the feature fired more in typed
        pairs_positive = sum(1 for d in diffs if d > 0)

        results.append({
            "feature_idx": int(feat_idx),
            "mean_paired_diff": round(mean_diff, 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": float(p_value),
            "cohens_d": round(cohens_d, 4),
            "n_pairs": len(diffs),
            "pairs_positive": pairs_positive,
            "pairs_positive_frac": round(pairs_positive / len(diffs), 4),
        })

    # Sort by t-statistic descending (strongest positive signal)
    results.sort(key=lambda x: x["t_statistic"], reverse=True)
    return results[:top_n]


def print_table(features: list[dict], title: str, columns: list[tuple[str, str, int]]):
    """Print a formatted summary table."""
    print(f"\n{'='*100}")
    print(title)
    print(f"{'='*100}")

    header = ""
    for col_name, _, width in columns:
        header += f"{col_name:<{width}} "
    print(header)
    print("-" * 100)

    for i, feat in enumerate(features[:30], 1):
        row = ""
        for _, key, width in columns:
            if key == "__rank__":
                row += f"{i:<{width}} "
            else:
                val = feat.get(key, "")
                if isinstance(val, float):
                    row += f"{val:<{width}.4f} "
                else:
                    row += f"{val:<{width}} "
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Find typing-specific SAE features")
    parser.add_argument("--top-n", type=int, default=50, help="Number of top features per mode")
    parser.add_argument("--min-freq", type=float, default=0.03, help="Minimum typed frequency threshold")
    parser.add_argument("--results-dir", type=str, default="scripts/typing/results", help="Results directory")
    parser.add_argument("--feature-key", type=str, default="top_features",
                        choices=["top_features", "mean_pooled_features"],
                        help="Which activation type to analyze")
    args = parser.parse_args()

    typed_path = os.path.join(args.results_dir, "activations_typed.json")
    untyped_path = os.path.join(args.results_dir, "activations_untyped.json")

    if not os.path.exists(typed_path) or not os.path.exists(untyped_path):
        print("Error: activation files not found. Run 01_collect_activations.py first.")
        return

    print(f"Loading activation data (feature_key={args.feature_key})...")
    typed_data = load_activations(typed_path)
    untyped_data = load_activations(untyped_path)

    # Filter out error entries
    typed_data = [r for r in typed_data if r.get(args.feature_key)]
    untyped_data = [r for r in untyped_data if r.get(args.feature_key)]
    print(f"  Typed prompts with features: {len(typed_data)}")
    print(f"  Untyped prompts with features: {len(untyped_data)}")

    # ── Mode 1: Combined Ranking ──
    print("\n[Mode 1] Combined differential ranking...")
    typed_stats = compute_feature_stats(typed_data, args.feature_key)
    untyped_stats = compute_feature_stats(untyped_data, args.feature_key)

    combined_ranked = rank_features(typed_stats, untyped_stats, args.min_freq, args.top_n)
    print_table(combined_ranked, f"Mode 1: Top {len(combined_ranked)} Typing Features (Combined)", [
        ("Rank", "__rank__", 5),
        ("Feature", "feature_idx", 10),
        ("Score", "differential_score", 10),
        ("T.Freq", "typed_frequency", 8),
        ("T.Act", "typed_mean_activation", 8),
        ("U.Freq", "untyped_frequency", 8),
        ("U.Act", "untyped_mean_activation", 8),
        ("Ratio", "specificity_ratio", 8),
    ])

    # ── Mode 2: Per-Language-Family Ranking ──
    print("\n[Mode 2] Per-language-family ranking...")

    ts_typed = [r for r in typed_data if r["language_family"] == "ts_js"]
    ts_untyped = [r for r in untyped_data if r["language_family"] == "ts_js"]
    py_typed = [r for r in typed_data if r["language_family"] == "python"]
    py_untyped = [r for r in untyped_data if r["language_family"] == "python"]

    ts_stats_t = compute_feature_stats(ts_typed, args.feature_key)
    ts_stats_u = compute_feature_stats(ts_untyped, args.feature_key)
    ts_ranked = rank_features(ts_stats_t, ts_stats_u, args.min_freq, args.top_n)

    py_stats_t = compute_feature_stats(py_typed, args.feature_key)
    py_stats_u = compute_feature_stats(py_untyped, args.feature_key)
    py_ranked = rank_features(py_stats_t, py_stats_u, args.min_freq, args.top_n)

    print_table(ts_ranked[:20], "Mode 2a: Top 20 TS/JS Typing Features", [
        ("Rank", "__rank__", 5),
        ("Feature", "feature_idx", 10),
        ("Score", "differential_score", 10),
        ("T.Freq", "typed_frequency", 8),
        ("T.Act", "typed_mean_activation", 8),
        ("U.Freq", "untyped_frequency", 8),
        ("U.Act", "untyped_mean_activation", 8),
    ])

    print_table(py_ranked[:20], "Mode 2b: Top 20 Python Typing Features", [
        ("Rank", "__rank__", 5),
        ("Feature", "feature_idx", 10),
        ("Score", "differential_score", 10),
        ("T.Freq", "typed_frequency", 8),
        ("T.Act", "typed_mean_activation", 8),
        ("U.Freq", "untyped_frequency", 8),
        ("U.Act", "untyped_mean_activation", 8),
    ])

    # Cross-language consensus: features in top 20 of BOTH families
    ts_top20 = {f["feature_idx"] for f in ts_ranked[:20]}
    py_top20 = {f["feature_idx"] for f in py_ranked[:20]}
    cross_lang_consensus = ts_top20 & py_top20
    print(f"\nCross-language consensus (top 20 in both TS/JS AND Python): {sorted(cross_lang_consensus)}")

    # ── Mode 3: Paired Difference Analysis ──
    print("\n[Mode 3] Paired difference analysis (t-test)...")
    paired_results = paired_analysis(typed_data, untyped_data, args.feature_key, args.top_n)

    # Filter to significant results
    significant = [r for r in paired_results if r["p_value"] < 0.05]
    print(f"  Features with p < 0.05: {len(significant)}/{len(paired_results)}")

    print_table(paired_results[:30], "Mode 3: Top 30 Features by Paired T-Test", [
        ("Rank", "__rank__", 5),
        ("Feature", "feature_idx", 10),
        ("MeanDiff", "mean_paired_diff", 10),
        ("t-stat", "t_statistic", 8),
        ("p-value", "p_value", 10),
        ("Cohen-d", "cohens_d", 8),
        ("PairPos", "pairs_positive_frac", 8),
    ])

    # ── Consensus Features ──
    combined_top50 = {f["feature_idx"] for f in combined_ranked[:50]}
    paired_sig = {f["feature_idx"] for f in paired_results if f["p_value"] < 0.05}

    consensus = combined_top50 & cross_lang_consensus & paired_sig
    print(f"\n{'='*60}")
    print(f"CONSENSUS FEATURES (top 50 combined + top 20 both families + paired p<0.05)")
    print(f"{'='*60}")
    print(f"  Features: {sorted(consensus)}")
    print(f"  Count: {len(consensus)}")

    # Build detailed consensus list
    consensus_details = []
    for feat_idx in sorted(consensus):
        combined_entry = next((f for f in combined_ranked if f["feature_idx"] == feat_idx), {})
        paired_entry = next((f for f in paired_results if f["feature_idx"] == feat_idx), {})
        ts_entry = next((f for f in ts_ranked if f["feature_idx"] == feat_idx), {})
        py_entry = next((f for f in py_ranked if f["feature_idx"] == feat_idx), {})

        consensus_details.append({
            "feature_idx": feat_idx,
            "combined_score": combined_entry.get("differential_score", 0),
            "combined_specificity": combined_entry.get("specificity_ratio", 0),
            "ts_score": ts_entry.get("differential_score", 0),
            "py_score": py_entry.get("differential_score", 0),
            "paired_mean_diff": paired_entry.get("mean_paired_diff", 0),
            "paired_p_value": paired_entry.get("p_value", 1),
            "paired_cohens_d": paired_entry.get("cohens_d", 0),
        })

    # ── Save all results ──
    output = {
        "analysis_config": {
            "feature_key": args.feature_key,
            "min_freq": args.min_freq,
            "top_n": args.top_n,
            "n_typed": len(typed_data),
            "n_untyped": len(untyped_data),
        },
        "mode1_combined": combined_ranked,
        "mode2_ts_js": ts_ranked,
        "mode2_python": py_ranked,
        "mode2_cross_language_consensus": sorted(cross_lang_consensus),
        "mode3_paired": paired_results,
        "mode3_significant_count": len(significant),
        "consensus_features": consensus_details,
        "consensus_feature_indices": sorted(consensus),
    }

    output_path = os.path.join(args.results_dir, "typing_features_ranked.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved full analysis to {output_path}")


if __name__ == "__main__":
    main()
