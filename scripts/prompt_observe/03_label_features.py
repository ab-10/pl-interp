#!/usr/bin/env python3
"""Label top features using contrastive target/control outputs via Mistral API.

For each top feature per property, shows Mistral the generated outputs where
the feature fired most strongly (target) alongside outputs where it fired
weakly or not at all (control), enabling contrastive interpretation.

Requires MISTRAL_API_KEY environment variable.

Usage:
    python 03_label_features.py
    python 03_label_features.py --top-n 10 --examples-per-feature 3
"""

import argparse
import json
import os
import re
import sys
import time

from mistralai import Mistral
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from properties import PROPERTIES


def load_ranked(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_activations(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def find_contrastive_examples(
    feature_idx: int,
    target_results: list[dict],
    control_results: list[dict],
    feature_key: str,
    n: int,
) -> list[dict]:
    """Find target outputs where the feature fires strongly and matching
    control outputs where it fires weakly, forming contrastive pairs."""

    # Find target outputs with high activation for this feature
    target_hits = []
    for result in target_results:
        for feat in result.get(feature_key, []):
            if feat["feature_idx"] == feature_idx:
                target_hits.append({
                    "prompt": result["prompt"],
                    "output": result["output"],
                    "activation": feat["activation"],
                    "role": "target",
                })
                break

    target_hits.sort(key=lambda x: x["activation"], reverse=True)

    # Find control outputs with low or zero activation for this feature
    control_hits = []
    for result in control_results:
        act = 0.0
        for feat in result.get(feature_key, []):
            if feat["feature_idx"] == feature_idx:
                act = feat["activation"]
                break
        control_hits.append({
            "prompt": result["prompt"],
            "output": result["output"],
            "activation": act,
            "role": "control",
        })

    # Sort control by activation ascending (lowest first = best contrast)
    control_hits.sort(key=lambda x: x["activation"])

    # Build contrastive pairs: pair each top target with the lowest control
    pairs = []
    for i, target in enumerate(target_hits[:n]):
        control = control_hits[i] if i < len(control_hits) else control_hits[-1]
        pairs.append({
            "target_prompt": target["prompt"],
            "target_output": target["output"][:500],
            "target_activation": target["activation"],
            "control_prompt": control["prompt"],
            "control_output": control["output"][:500],
            "control_activation": control["activation"],
        })

    return pairs


def label_feature(
    client: Mistral,
    feature_idx: int,
    property_name: str,
    contrastive_pairs: list[dict],
    model: str,
) -> dict:
    """Send contrastive pairs to Mistral API for labeling."""
    pairs_text = ""
    for i, pair in enumerate(contrastive_pairs, 1):
        pairs_text += f"""
--- Pair {i} ---

HIGH activation ({pair['target_activation']:.2f}) — prompted with: "{pair['target_prompt'][:100]}"
Generated code:
{pair['target_output'][:400]}

LOW activation ({pair['control_activation']:.2f}) — prompted with: "{pair['control_prompt'][:100]}"
Generated code:
{pair['control_output'][:400]}
"""

    prompt = f"""Below are contrastive pairs of model-generated code. In each pair, the FIRST
output strongly activates feature #{feature_idx} of a language model, while the
SECOND output activates it weakly or not at all.

These were generated while investigating the coding property "{property_name}".

{pairs_text}

Your job: identify what specific code pattern or property the HIGH-activation
outputs have that the LOW-activation outputs lack. Focus on the DIFFERENCES
between each pair's generated code (not the prompts).

Provide:
1. LABEL: A short label (3-7 words) describing what this feature detects
2. DESCRIPTION: A one-sentence description of the feature's apparent function
3. PROPERTY_MATCH: Does this feature clearly correspond to "{property_name}"? (yes/partial/no)

Format your response exactly as:
LABEL: <your label>
DESCRIPTION: <your description>
PROPERTY_MATCH: <yes/partial/no>"""

    response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )

    response_text = response.choices[0].message.content.strip()

    label = ""
    description = ""
    property_match = ""

    label_match = re.search(r"LABEL:\s*(.+?)(?:\n|$)", response_text)
    desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?:\n|$)", response_text)
    match_match = re.search(r"PROPERTY_MATCH:\s*(.+?)(?:\n|$)", response_text)

    if label_match:
        label = label_match.group(1).strip()
    if desc_match:
        description = desc_match.group(1).strip()
    if match_match:
        property_match = match_match.group(1).strip().lower()

    return {
        "label": label,
        "description": description,
        "property_match": property_match,
        "raw_response": response_text,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Label features with Mistral API using contrastive examples"
    )
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top features to label per property")
    parser.add_argument("--examples-per-feature", type=int, default=5,
                        help="Contrastive pairs per feature")
    parser.add_argument("--model", type=str, default="mistral-medium-latest",
                        help="Mistral model to use")
    parser.add_argument("--feature-key", type=str, default="mean_pooled_features",
                        choices=["mean_pooled_features", "last_token_features"],
                        help="Which activation type was analyzed")
    parser.add_argument("--results-dir", type=str,
                        default="scripts/prompt_observe/results",
                        help="Results directory")
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set")
        return

    # Load ranked features
    ranked_path = os.path.join(args.results_dir, "features_ranked.json")
    if not os.path.exists(ranked_path):
        print("Error: features_ranked.json not found. Run 02_rank_features.py first.")
        return

    ranked = load_ranked(ranked_path)
    per_property = ranked.get("per_property", {})

    if not per_property:
        print("Error: No per-property rankings found in features_ranked.json")
        return

    client = Mistral(api_key=api_key)
    registry = []

    for prop_name, features in per_property.items():
        # Load the activation data for this property
        act_path = os.path.join(args.results_dir, f"activations_{prop_name}.json")
        if not os.path.exists(act_path):
            print(f"Warning: {act_path} not found, skipping {prop_name}")
            continue

        act_data = load_activations(act_path)
        features_to_label = features[:args.top_n]

        print(f"\n{'='*70}")
        print(f"Labeling {len(features_to_label)} features for: {prop_name}")
        print(f"{'='*70}")

        for feat in tqdm(features_to_label, desc=f"Labeling {prop_name}"):
            feature_idx = feat["feature_idx"]

            contrastive_pairs = find_contrastive_examples(
                feature_idx,
                act_data["target_results"],
                act_data["control_results"],
                args.feature_key,
                args.examples_per_feature,
            )

            if not contrastive_pairs:
                print(f"  Warning: No contrastive pairs for feature {feature_idx}")
                registry.append({
                    "feature_idx": feature_idx,
                    "property": prop_name,
                    "label": "unknown",
                    "description": "No contrastive pairs found",
                    "property_match": "unknown",
                    "differential_score": feat.get("differential_score", 0),
                    "examples": [],
                })
                continue

            try:
                labeling = label_feature(
                    client, feature_idx, prop_name,
                    contrastive_pairs, args.model,
                )
                print(f"  Feature {feature_idx}: {labeling['label']} "
                      f"(match={labeling['property_match']})")

                registry.append({
                    "feature_idx": feature_idx,
                    "property": prop_name,
                    "label": labeling["label"],
                    "description": labeling["description"],
                    "property_match": labeling["property_match"],
                    "raw_response": labeling["raw_response"],
                    "differential_score": feat.get("differential_score", 0),
                    "target_frequency": feat.get("target_frequency", 0),
                    "specificity_ratio": feat.get("specificity_ratio", 0),
                    "examples": [
                        {
                            "target_prompt": p["target_prompt"][:100],
                            "target_activation": p["target_activation"],
                            "control_activation": p["control_activation"],
                        }
                        for p in contrastive_pairs
                    ],
                })

            except Exception as e:
                print(f"  Error labeling feature {feature_idx}: {e}")
                registry.append({
                    "feature_idx": feature_idx,
                    "property": prop_name,
                    "label": "error",
                    "description": str(e),
                    "property_match": "unknown",
                    "differential_score": feat.get("differential_score", 0),
                    "examples": [],
                })

            time.sleep(1)  # Rate limit

    # Save registry
    output_path = os.path.join(args.results_dir, "feature_registry.json")
    with open(output_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"\nSaved {len(registry)} labeled features to {output_path}")

    # Print summary
    print(f"\n{'='*90}")
    print("Feature Registry Summary")
    print(f"{'='*90}")
    print(f"{'Feature':<10} {'Property':<22} {'Score':<10} {'Match':<9} {'Label'}")
    print("-" * 90)
    for entry in registry:
        print(
            f"{entry['feature_idx']:<10} "
            f"{entry['property']:<22} "
            f"{entry.get('differential_score', 0):<10.4f} "
            f"{entry.get('property_match', '?'):<9} "
            f"{entry['label']}"
        )

    # Count matches
    yes_count = sum(1 for e in registry if e.get("property_match") == "yes")
    partial_count = sum(1 for e in registry if e.get("property_match") == "partial")
    no_count = sum(1 for e in registry if e.get("property_match") == "no")
    print(f"\nProperty match: {yes_count} yes, {partial_count} partial, {no_count} no")


if __name__ == "__main__":
    main()
