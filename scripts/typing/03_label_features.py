#!/usr/bin/env python3
"""Label top typing features using Mistral API with contrastive pairs.

For each top feature, shows Mistral the typed snippet (high activation)
alongside its untyped counterpart (low activation) to give direct evidence
of what the feature responds to.

Requires MISTRAL_API_KEY environment variable.

Usage:
    python 03_label_features.py [--top-n 20] [--examples-per-feature 5]
"""

import argparse
import json
import os
import re
import time

from mistralai import Mistral
from tqdm import tqdm


def load_ranked_features(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_activations(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def get_contrastive_pairs(
    feature_idx: int,
    typed_data: list[dict],
    untyped_data: list[dict],
    n: int,
    feature_key: str = "top_features",
) -> list[dict]:
    """Find pairs where the feature fires strongly in typed but weakly in untyped."""
    # Build untyped lookup by pair_id
    untyped_by_pair = {}
    for record in untyped_data:
        untyped_by_pair[record["pair_id"]] = record

    pairs = []
    for record in typed_data:
        typed_act = 0.0
        for feat in record.get(feature_key, []):
            if feat["feature_idx"] == feature_idx:
                typed_act = feat["activation"]
                break

        if typed_act <= 0:
            continue

        pair_id = record["pair_id"]
        untyped_record = untyped_by_pair.get(pair_id)
        if not untyped_record:
            continue

        untyped_act = 0.0
        for feat in untyped_record.get(feature_key, []):
            if feat["feature_idx"] == feature_idx:
                untyped_act = feat["activation"]
                break

        pairs.append({
            "pair_id": pair_id,
            "category": record["category"],
            "language": record["language"],
            "typed_text": record["text"],
            "typed_activation": typed_act,
            "untyped_text": untyped_record["text"],
            "untyped_activation": untyped_act,
            "activation_diff": typed_act - untyped_act,
        })

    # Sort by activation difference (biggest difference first)
    pairs.sort(key=lambda x: x["activation_diff"], reverse=True)
    return pairs[:n]


def label_feature(
    client: Mistral,
    feature_idx: int,
    contrastive_pairs: list[dict],
    model: str,
) -> dict:
    """Send contrastive pairs to Mistral API for labeling."""
    pairs_text = ""
    for i, pair in enumerate(contrastive_pairs, 1):
        pairs_text += f"""
--- Pair {i} ({pair['language']}, {pair['category']}) ---

HIGH activation ({pair['typed_activation']:.2f}) version:
{pair['typed_text']}

LOW activation ({pair['untyped_activation']:.2f}) version:
{pair['untyped_text']}
"""

    prompt = f"""Below are contrastive pairs of code snippets. In each pair, the FIRST version
strongly activates feature #{feature_idx} of a language model, while the SECOND
version activates it weakly or not at all. The pairs are the SAME code, but
one version has something the other lacks.

{pairs_text}

Your job: identify what specific code property the HIGH-activation versions have
that the LOW-activation versions lack. Focus on the DIFFERENCES between each pair.

Provide:
1. LABEL: A short label (3-7 words) describing what this feature detects
2. DESCRIPTION: A one-sentence description of the feature's apparent function
3. SPECIFICITY: Is this feature language-specific or cross-language? (one of: typescript-specific, python-specific, cross-language)

Format your response exactly as:
LABEL: <your label>
DESCRIPTION: <your description>
SPECIFICITY: <specificity>"""

    response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )

    response_text = response.choices[0].message.content.strip()

    label = ""
    description = ""
    specificity = ""

    label_match = re.search(r"LABEL:\s*(.+?)(?:\n|$)", response_text)
    desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?:\n|$)", response_text)
    spec_match = re.search(r"SPECIFICITY:\s*(.+?)(?:\n|$)", response_text)

    if label_match:
        label = label_match.group(1).strip()
    if desc_match:
        description = desc_match.group(1).strip()
    if spec_match:
        specificity = spec_match.group(1).strip()

    return {
        "label": label,
        "description": description,
        "specificity": specificity,
        "raw_response": response_text,
    }


def main():
    parser = argparse.ArgumentParser(description="Label typing features with Mistral API")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top features to label")
    parser.add_argument("--examples-per-feature", type=int, default=5, help="Contrastive pairs per feature")
    parser.add_argument("--model", type=str, default="mistral-medium-latest", help="Mistral model to use")
    parser.add_argument("--results-dir", type=str, default="scripts/typing/results", help="Results directory")
    parser.add_argument("--feature-key", type=str, default="top_features",
                        choices=["top_features", "mean_pooled_features"],
                        help="Which activation type was analyzed")
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set")
        return

    features_path = os.path.join(args.results_dir, "typing_features_ranked.json")
    typed_path = os.path.join(args.results_dir, "activations_typed.json")
    untyped_path = os.path.join(args.results_dir, "activations_untyped.json")

    if not os.path.exists(features_path):
        print("Error: typing_features_ranked.json not found. Run 02_find_typing_features.py first.")
        return

    print("Loading ranked features and activation data...")
    analysis = load_ranked_features(features_path)
    typed_data = load_activations(typed_path)
    untyped_data = load_activations(untyped_path)

    # Use consensus features first, then fill from combined ranking
    consensus_indices = set(analysis.get("consensus_feature_indices", []))
    combined_features = analysis.get("mode1_combined", [])

    # Build ordered list: consensus first, then remaining from combined
    features_to_label = []
    for feat in combined_features:
        if feat["feature_idx"] in consensus_indices:
            features_to_label.append({**feat, "is_consensus": True})

    for feat in combined_features:
        if feat["feature_idx"] not in consensus_indices:
            features_to_label.append({**feat, "is_consensus": False})

    features_to_label = features_to_label[:args.top_n]
    print(f"Will label {len(features_to_label)} features ({sum(1 for f in features_to_label if f.get('is_consensus'))} consensus)")

    client = Mistral(api_key=api_key)

    registry = []
    for i, feat in enumerate(tqdm(features_to_label, desc="Labeling features")):
        feature_idx = feat["feature_idx"]
        print(f"\n[{i+1}/{len(features_to_label)}] Labeling feature {feature_idx} "
              f"({'CONSENSUS' if feat.get('is_consensus') else 'combined'})...")

        contrastive_pairs = get_contrastive_pairs(
            feature_idx, typed_data, untyped_data,
            args.examples_per_feature, args.feature_key
        )

        if not contrastive_pairs:
            print(f"  Warning: No contrastive pairs found for feature {feature_idx}")
            registry.append({
                "feature_idx": feature_idx,
                "label": "unknown",
                "description": "No contrastive pairs found",
                "specificity": "unknown",
                "is_consensus": feat.get("is_consensus", False),
                "differential_score": feat.get("differential_score", 0),
                "examples": [],
            })
            continue

        try:
            labeling = label_feature(client, feature_idx, contrastive_pairs, args.model)
            print(f"  Label: {labeling['label']}")
            print(f"  Description: {labeling['description']}")
            print(f"  Specificity: {labeling['specificity']}")

            registry.append({
                "feature_idx": feature_idx,
                "label": labeling["label"],
                "description": labeling["description"],
                "specificity": labeling["specificity"],
                "raw_response": labeling["raw_response"],
                "is_consensus": feat.get("is_consensus", False),
                "differential_score": feat.get("differential_score", 0),
                "typed_frequency": feat.get("typed_frequency", 0),
                "typed_mean_activation": feat.get("typed_mean_activation", 0),
                "examples": [
                    {
                        "pair_id": p["pair_id"],
                        "category": p["category"],
                        "language": p["language"],
                        "typed_activation": p["typed_activation"],
                        "untyped_activation": p["untyped_activation"],
                        "typed_text": p["typed_text"][:300],
                        "untyped_text": p["untyped_text"][:300],
                    }
                    for p in contrastive_pairs
                ],
            })

        except Exception as e:
            print(f"  Error labeling feature {feature_idx}: {e}")
            registry.append({
                "feature_idx": feature_idx,
                "label": "error",
                "description": str(e),
                "specificity": "unknown",
                "is_consensus": feat.get("is_consensus", False),
                "differential_score": feat.get("differential_score", 0),
                "examples": [],
            })

        time.sleep(1)

    # Save registry
    output_path = os.path.join(args.results_dir, "typing_feature_registry.json")
    with open(output_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"\nSaved {len(registry)} labeled features to {output_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("Typing Feature Registry Summary")
    print(f"{'='*80}")
    print(f"{'Idx':<10} {'Score':<10} {'Consensus':<10} {'Specificity':<20} {'Label':<40}")
    print("-" * 80)
    for entry in registry:
        print(
            f"{entry['feature_idx']:<10} "
            f"{entry.get('differential_score', 0):<10.4f} "
            f"{'YES' if entry.get('is_consensus') else 'no':<10} "
            f"{entry.get('specificity', 'unknown'):<20} "
            f"{entry['label']:<40}"
        )


if __name__ == "__main__":
    main()
