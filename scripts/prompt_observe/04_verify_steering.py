#!/usr/bin/env python3
"""Verify steering effect for labeled coding features.

Tests bidirectional steering for each labeled feature using ambiguous prompts
and property-specific density metrics. Positive steering should increase the
property's presence; negative steering should decrease it.

Usage:
    python 04_verify_steering.py
    python 04_verify_steering.py --max-features 5 --max-prompts 3
"""

import argparse
import json
import os
import sys

import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer

sys.path.insert(0, os.path.dirname(__file__))
from properties import PROPERTIES, compute_property_density


STRENGTHS = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]

# Ambiguous prompts that could go in any direction — the steering should
# push these toward or away from each property.
AMBIGUOUS_PROMPTS = [
    "Write a Python function that takes a list and returns the largest element",
    "Write a function that filters an array of objects by a property value",
    "Implement a stack data structure with push, pop, and peek methods",
    "Write a function that merges two sorted arrays into one sorted array",
    "Create a function that validates an email address",
    "Write a function that converts a nested dictionary to a flat dictionary",
    "Implement a binary search function",
    "Write a function that groups items by a key",
]


def generate_text(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate text without steering (baseline)."""
    tokens = model.to_tokens(prompt, prepend_bos=False)
    with torch.no_grad():
        output = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
    return model.tokenizer.decode(output[0], skip_special_tokens=True)


def generate_steered(
    model: HookedTransformer,
    sae: SAE,
    hook_point: str,
    prompt: str,
    feature_idx: int,
    strength: float,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate text with steering via SAE decoder vector."""
    steering_vector = sae.W_dec[feature_idx].detach().clone()

    def steering_hook(value, hook):
        value[:, :, :] = value + strength * steering_vector.to(
            value.device, value.dtype
        )
        return value

    tokens = model.to_tokens(prompt, prepend_bos=False)
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hook_point, steering_hook)]):
            output = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
    return model.tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Verify steering effect for coding features"
    )
    parser.add_argument("--max-tokens", type=int, default=300,
                        help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--max-features", type=int, default=None,
                        help="Max features to verify (per property)")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Max prompts to test per feature")
    parser.add_argument("--results-dir", type=str,
                        default="scripts/prompt_observe/results",
                        help="Results directory")
    args = parser.parse_args()

    registry_path = os.path.join(args.results_dir, "feature_registry.json")
    if not os.path.exists(registry_path):
        print("Error: feature_registry.json not found. Run 03_label_features.py first.")
        return

    print("Loading feature registry...")
    with open(registry_path) as f:
        registry = json.load(f)

    # Group features by property
    by_property = {}
    for entry in registry:
        prop = entry.get("property", "unknown")
        by_property.setdefault(prop, []).append(entry)

    if args.max_features:
        by_property = {
            k: v[:args.max_features] for k, v in by_property.items()
        }

    total_features = sum(len(v) for v in by_property.values())
    print(f"  Will verify {total_features} features across {len(by_property)} properties")

    prompts = AMBIGUOUS_PROMPTS
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    # Load model
    print("Loading Mistral 7B...")
    model = HookedTransformer.from_pretrained_no_processing(
        "mistralai/Mistral-7B-Instruct-v0.1",
        dtype=torch.float16,
    )
    print(f"  Model loaded on {model.cfg.device}")

    # Load SAE
    print("Loading SAE...")
    sae_tuple = SAE.from_pretrained(
        release="tylercosgrove/mistral-7b-sparse-autoencoder-layer16",
        sae_id=".",
    )
    sae = sae_tuple[0] if isinstance(sae_tuple, tuple) else sae_tuple
    sae = sae.to(model.cfg.device)

    hook_point = sae.cfg.metadata["hook_name"]
    print(f"  SAE loaded, hook point: {hook_point}")
    print(f"\nStrengths: {STRENGTHS}")
    print(f"Prompts: {len(prompts)}")
    print(f"Temperature: {args.temperature}")

    all_results = []

    for prop_name, features in by_property.items():
        print(f"\n{'#'*80}")
        print(f"Property: {prop_name}")
        print(f"{'#'*80}")

        # Check if this property has detection patterns
        has_metric = prop_name in PROPERTIES

        for feature in tqdm(features, desc=f"Verifying {prop_name}"):
            feature_idx = feature["feature_idx"]
            label = feature.get("label", "unknown")

            print(f"\n  Feature {feature_idx}: {label}")

            feature_result = {
                "feature_idx": feature_idx,
                "property": prop_name,
                "label": label,
                "description": feature.get("description", ""),
                "property_match": feature.get("property_match", "unknown"),
                "prompt_results": [],
            }

            for prompt in prompts:
                prompt_result = {
                    "prompt": prompt,
                    "strength_results": [],
                }

                for strength in STRENGTHS:
                    try:
                        if strength == 0.0:
                            output = generate_text(
                                model, prompt, args.max_tokens, args.temperature
                            )
                        else:
                            output = generate_steered(
                                model, sae, hook_point, prompt,
                                feature_idx, strength,
                                args.max_tokens, args.temperature,
                            )

                        result_entry = {
                            "strength": strength,
                            "output": output,
                            "status": "success",
                        }

                        # Compute property-specific density if available
                        if has_metric:
                            density = compute_property_density(output, prop_name)
                            result_entry["density"] = density["density"]
                            result_entry["total_markers"] = density["total_markers"]
                            result_entry["total_lines"] = density["total_lines"]

                        prompt_result["strength_results"].append(result_entry)

                    except Exception as e:
                        print(f"    strength={strength:+.1f}: ERROR {e}")
                        prompt_result["strength_results"].append({
                            "strength": strength,
                            "error": str(e),
                            "status": "error",
                        })

                    torch.cuda.empty_cache()

                feature_result["prompt_results"].append(prompt_result)

            # Compute summary: average density at each strength
            if has_metric:
                strength_densities = {s: [] for s in STRENGTHS}
                for pr in feature_result["prompt_results"]:
                    for sr in pr["strength_results"]:
                        if sr["status"] == "success" and "density" in sr:
                            strength_densities[sr["strength"]].append(sr["density"])

                feature_result["density_by_strength"] = {
                    str(s): round(sum(d) / len(d), 4) if d else None
                    for s, d in strength_densities.items()
                }

                # Check monotonicity: does density increase with strength?
                valid = [
                    (s, feature_result["density_by_strength"].get(str(s)))
                    for s in STRENGTHS
                ]
                valid = [(s, d) for s, d in valid if d is not None]

                if len(valid) >= 3:
                    neg_avg = (
                        sum(d for s, d in valid if s < 0)
                        / max(sum(1 for s, _ in valid if s < 0), 1)
                    )
                    pos_avg = (
                        sum(d for s, d in valid if s > 0)
                        / max(sum(1 for s, _ in valid if s > 0), 1)
                    )
                    baseline_d = next((d for s, d in valid if s == 0.0), 0)

                    feature_result["monotonic_trend"] = pos_avg > baseline_d > neg_avg
                    feature_result["pos_avg_density"] = round(pos_avg, 4)
                    feature_result["neg_avg_density"] = round(neg_avg, 4)
                    feature_result["baseline_density"] = round(baseline_d, 4)

                    trend = "MONOTONIC" if feature_result["monotonic_trend"] else "non-monotonic"
                    print(f"    Trend: {trend} (neg={neg_avg:.3f}, base={baseline_d:.3f}, pos={pos_avg:.3f})")

            all_results.append(feature_result)

    # Save results
    output_path = os.path.join(args.results_dir, "steering_verification.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved {len(all_results)} verification results to {output_path}")

    # Print summary table
    print(f"\n{'='*100}")
    print("Steering Verification Summary")
    print(f"{'='*100}")
    print(
        f"{'Feature':<10} {'Property':<22} {'Label':<25} {'Match':<9} "
        f"{'Mono':<7} {'NegAvg':<9} {'Base':<9} {'PosAvg':<9}"
    )
    print("-" * 100)

    validated = 0
    for r in all_results:
        mono = r.get("monotonic_trend")
        mono_str = "YES" if mono else ("no" if mono is not None else "N/A")
        if mono:
            validated += 1

        print(
            f"{r['feature_idx']:<10} "
            f"{r['property']:<22} "
            f"{r['label'][:23]:<25} "
            f"{r.get('property_match', '?'):<9} "
            f"{mono_str:<7} "
            f"{r.get('neg_avg_density', 0):<9.4f} "
            f"{r.get('baseline_density', 0):<9.4f} "
            f"{r.get('pos_avg_density', 0):<9.4f}"
        )

    total = len(all_results)
    print(f"\nValidated (monotonic trend): {validated}/{total}")

    # Fully validated: property_match=yes AND monotonic
    fully_validated = [
        r for r in all_results
        if r.get("property_match") == "yes" and r.get("monotonic_trend")
    ]
    if fully_validated:
        print(f"\nFully validated features (label matches property + monotonic steering):")
        for r in fully_validated:
            print(f"  Feature {r['feature_idx']} ({r['property']}): {r['label']}")


if __name__ == "__main__":
    main()
