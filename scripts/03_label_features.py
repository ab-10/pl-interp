#!/usr/bin/env python3
"""Label top code features using Mistral API.

For each top feature, collects the prompts that most strongly activated it,
sends them to Mistral for interpretation, and saves structured labels.

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


def load_ranked_features(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_activations(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def get_top_activating_prompts(
    feature_idx: int, activations: list[dict], n: int
) -> list[dict]:
    """Find the prompts where this feature fired most strongly."""
    prompt_acts = []
    for record in activations:
        for feat in record.get("top_features", []):
            if feat["feature_idx"] == feature_idx:
                prompt_acts.append({
                    "text": record["text"],
                    "id": record["id"],
                    "activation": feat["activation"],
                })
                break

    # Sort by activation strength, return top N
    prompt_acts.sort(key=lambda x: x["activation"], reverse=True)
    return prompt_acts[:n]


def label_feature(
    client: Mistral,
    feature_idx: int,
    examples: list[dict],
    model: str,
) -> dict:
    """Send activating examples to Mistral API for labeling."""
    examples_text = "\n\n---\n\n".join(
        f"Example {i+1} (activation: {ex['activation']:.2f}):\n{ex['text']}"
        for i, ex in enumerate(examples)
    )

    prompt = f"""Below are text snippets that all strongly activate a specific internal feature
(feature #{feature_idx}) of a language model. Your job is to identify what concept,
pattern, or property these snippets share that causes this feature to fire.

{examples_text}

Based on the common patterns across these examples, provide:
1. LABEL: A short label (3-7 words) describing what this feature detects
2. DESCRIPTION: A one-sentence description of the feature's apparent function

Format your response exactly as:
LABEL: <your label>
DESCRIPTION: <your description>"""

    response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3,
    )

    response_text = response.choices[0].message.content.strip()

    # Parse LABEL and DESCRIPTION from response
    label = ""
    description = ""

    label_match = re.search(r"LABEL:\s*(.+?)(?:\n|$)", response_text)
    desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?:\n|$)", response_text)

    if label_match:
        label = label_match.group(1).strip()
    if desc_match:
        description = desc_match.group(1).strip()

    return {
        "label": label,
        "description": description,
        "raw_response": response_text,
    }


def main():
    parser = argparse.ArgumentParser(description="Label code features with Mistral API")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top features to label")
    parser.add_argument("--examples-per-feature", type=int, default=5, help="Activating examples per feature")
    parser.add_argument("--model", type=str, default="mistral-medium-latest", help="Mistral model to use")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory")
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set")
        return

    features_path = os.path.join(args.results_dir, "code_features_ranked.json")
    code_acts_path = os.path.join(args.results_dir, "activations_code.json")

    if not os.path.exists(features_path):
        print("Error: code_features_ranked.json not found. Run 02_find_code_features.py first.")
        return

    # Load data
    print("Loading ranked features and activation data...")
    ranked_features = load_ranked_features(features_path)
    code_activations = load_activations(code_acts_path)

    features_to_label = ranked_features[: args.top_n]
    print(f"Will label top {len(features_to_label)} features")

    # Initialize Mistral client
    client = Mistral(api_key=api_key)

    # Label each feature
    registry = []
    for i, feat in enumerate(tqdm(features_to_label, desc="Labeling features")):
        feature_idx = feat["feature_idx"]
        print(f"\n[{i+1}/{len(features_to_label)}] Labeling feature {feature_idx}...")

        # Get top activating examples
        examples = get_top_activating_prompts(
            feature_idx, code_activations, args.examples_per_feature
        )

        if not examples:
            print(f"  Warning: No activating examples found for feature {feature_idx}")
            registry.append({
                "feature_idx": feature_idx,
                "label": "unknown",
                "description": "No activating examples found",
                "differential_score": feat["differential_score"],
                "code_frequency": feat["code_frequency"],
                "examples": [],
            })
            continue

        try:
            labeling = label_feature(client, feature_idx, examples, args.model)
            print(f"  Label: {labeling['label']}")
            print(f"  Description: {labeling['description']}")

            registry.append({
                "feature_idx": feature_idx,
                "label": labeling["label"],
                "description": labeling["description"],
                "raw_response": labeling["raw_response"],
                "differential_score": feat["differential_score"],
                "code_frequency": feat["code_frequency"],
                "code_mean_activation": feat["code_mean_activation"],
                "examples": [
                    {"id": ex["id"], "activation": ex["activation"], "text": ex["text"][:200]}
                    for ex in examples
                ],
            })

        except Exception as e:
            print(f"  Error labeling feature {feature_idx}: {e}")
            registry.append({
                "feature_idx": feature_idx,
                "label": "error",
                "description": str(e),
                "differential_score": feat["differential_score"],
                "code_frequency": feat["code_frequency"],
                "examples": [],
            })

        # Rate limit: 1 second between API calls
        time.sleep(1)

    # Save registry
    output_path = os.path.join(args.results_dir, "feature_registry.json")
    with open(output_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"\nSaved {len(registry)} labeled features to {output_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("Feature Registry Summary")
    print(f"{'='*70}")
    print(f"{'Idx':<10} {'Score':<10} {'Label':<50}")
    print("-" * 70)
    for entry in registry:
        print(
            f"{entry['feature_idx']:<10} "
            f"{entry['differential_score']:<10.4f} "
            f"{entry['label']:<50}"
        )


# Need tqdm import at top level for the progress bar
from tqdm import tqdm

if __name__ == "__main__":
    main()
