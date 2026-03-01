#!/usr/bin/env python3
"""Verify steering effect for labeled typing features.

Tests bidirectional steering strength sweep: positive should add types,
negative should remove them. Uses ambiguous prompts and computes a
typing_density metric to quantify the effect.

Usage:
    python 04_verify_steering.py [--max-tokens 300] [--temperature 0.3]
"""

import argparse
import json
import os
import re

import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer


STRENGTHS = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]

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

# Regex patterns for type annotation markers
TYPING_PATTERNS = [
    # Python type hints
    r":\s*(?:int|float|str|bool|list|dict|tuple|set|None|Any|Optional|Union|Callable|Iterator|Generator|Sequence|Mapping)\b",
    r"->\s*(?:int|float|str|bool|list|dict|tuple|set|None|Any|Optional|Union|Callable|Iterator|Generator|Sequence|Mapping)\b",
    r":\s*(?:List|Dict|Tuple|Set|FrozenSet|Deque|DefaultDict|OrderedDict|Counter)\[",
    r"\bTypeVar\b",
    r"\bGeneric\[",
    r"\bTypedDict\b",
    # TypeScript / general typed language markers
    r":\s*(?:number|string|boolean|void|never|any|unknown|undefined)\b",
    r":\s*(?:Array|Map|Set|Record|Promise|Partial|Required|Readonly)\s*<",
    r"<[A-Z]\w*(?:\s*,\s*[A-Z]\w*)*>",  # Generic type parameters
    r"\binterface\s+\w+",
    r"\btype\s+\w+\s*=",
    r":\s*\([^)]*\)\s*=>",  # Arrow function types
    r"\bas\s+\w+",  # Type assertions
]


def compute_typing_density(text: str) -> dict:
    """Count type annotation markers and compute typing density."""
    lines = text.strip().split("\n")
    total_lines = max(len(lines), 1)

    total_markers = 0
    marker_counts = {}

    for pattern in TYPING_PATTERNS:
        matches = re.findall(pattern, text)
        count = len(matches)
        if count > 0:
            marker_counts[pattern[:30]] = count
            total_markers += count

    return {
        "typing_density": round(total_markers / total_lines, 4),
        "total_markers": total_markers,
        "total_lines": total_lines,
        "marker_counts": marker_counts,
    }


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
        value[:, :, :] = value + strength * steering_vector.to(value.device, value.dtype)
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
    parser = argparse.ArgumentParser(description="Verify typing feature steering")
    parser.add_argument("--max-tokens", type=int, default=300, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--max-features", type=int, default=None, help="Max features to verify")
    parser.add_argument("--max-prompts", type=int, default=None, help="Max prompts to test per feature")
    parser.add_argument("--results-dir", type=str, default="scripts/typing/results", help="Results directory")
    args = parser.parse_args()

    registry_path = os.path.join(args.results_dir, "typing_feature_registry.json")
    if not os.path.exists(registry_path):
        print("Error: typing_feature_registry.json not found. Run 03_label_features.py first.")
        return

    print("Loading feature registry...")
    registry = load_registry(registry_path)
    if args.max_features:
        registry = registry[:args.max_features]
    print(f"  Will verify {len(registry)} features")

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
    sae = SAE.from_pretrained(
        release="tylercosgrove/mistral-7b-sparse-autoencoder-layer16",
        sae_id=".",
    )
    sae = sae.to(model.cfg.device)

    hook_point = sae.cfg.metadata["hook_name"]
    print(f"  SAE loaded, hook point: {hook_point}")

    print(f"\nStrengths to test: {STRENGTHS}")
    print(f"Prompts: {len(prompts)}")
    print(f"Temperature: {args.temperature}")

    # Verify each feature
    all_results = []

    for feature in tqdm(registry, desc="Verifying features"):
        feature_idx = feature["feature_idx"]
        label = feature.get("label", "unknown")

        print(f"\n{'='*80}")
        print(f"Feature {feature_idx}: {label}")
        print(f"{'='*80}")

        feature_result = {
            "feature_idx": feature_idx,
            "label": label,
            "description": feature.get("description", ""),
            "is_consensus": feature.get("is_consensus", False),
            "specificity": feature.get("specificity", "unknown"),
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

                    density = compute_typing_density(output)

                    prompt_result["strength_results"].append({
                        "strength": strength,
                        "output": output,
                        "typing_density": density["typing_density"],
                        "total_markers": density["total_markers"],
                        "total_lines": density["total_lines"],
                        "status": "success",
                    })

                    print(f"  strength={strength:+.1f}: density={density['typing_density']:.3f} "
                          f"(markers={density['total_markers']}, lines={density['total_lines']})")

                except Exception as e:
                    print(f"  strength={strength:+.1f}: ERROR {e}")
                    prompt_result["strength_results"].append({
                        "strength": strength,
                        "error": str(e),
                        "status": "error",
                    })

                torch.cuda.empty_cache()

            feature_result["prompt_results"].append(prompt_result)

        # Compute summary: average density at each strength across all prompts
        strength_densities = {s: [] for s in STRENGTHS}
        for pr in feature_result["prompt_results"]:
            for sr in pr["strength_results"]:
                if sr["status"] == "success":
                    strength_densities[sr["strength"]].append(sr["typing_density"])

        feature_result["density_by_strength"] = {
            str(s): round(sum(d) / len(d), 4) if d else None
            for s, d in strength_densities.items()
        }

        # Check monotonicity: does density increase with strength?
        densities_at_strengths = [
            (s, feature_result["density_by_strength"].get(str(s)))
            for s in STRENGTHS
        ]
        valid_densities = [(s, d) for s, d in densities_at_strengths if d is not None]

        if len(valid_densities) >= 3:
            # Check if positive strengths produce higher density than negative
            neg_avg = sum(d for s, d in valid_densities if s < 0) / max(sum(1 for s, _ in valid_densities if s < 0), 1)
            pos_avg = sum(d for s, d in valid_densities if s > 0) / max(sum(1 for s, _ in valid_densities if s > 0), 1)
            baseline_d = next((d for s, d in valid_densities if s == 0.0), 0)

            feature_result["monotonic_trend"] = pos_avg > baseline_d > neg_avg
            feature_result["pos_avg_density"] = round(pos_avg, 4)
            feature_result["neg_avg_density"] = round(neg_avg, 4)
            feature_result["baseline_density"] = round(baseline_d, 4)

            trend = "MONOTONIC" if feature_result["monotonic_trend"] else "non-monotonic"
            print(f"\n  Trend: {trend} (neg={neg_avg:.3f}, base={baseline_d:.3f}, pos={pos_avg:.3f})")
        else:
            feature_result["monotonic_trend"] = None

        all_results.append(feature_result)

    # Save results
    output_path = os.path.join(args.results_dir, "steering_verification.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved {len(all_results)} verification results to {output_path}")

    # Print summary table
    print(f"\n{'='*90}")
    print("Steering Verification Summary")
    print(f"{'='*90}")
    print(f"{'Feature':<10} {'Label':<30} {'Consensus':<10} {'Monotonic':<10} "
          f"{'Neg Avg':<10} {'Base':<10} {'Pos Avg':<10}")
    print("-" * 90)

    validated = 0
    for r in all_results:
        mono = r.get("monotonic_trend")
        mono_str = "YES" if mono else ("no" if mono is not None else "N/A")
        if mono:
            validated += 1

        print(
            f"{r['feature_idx']:<10} "
            f"{r['label'][:28]:<30} "
            f"{'YES' if r.get('is_consensus') else 'no':<10} "
            f"{mono_str:<10} "
            f"{r.get('neg_avg_density', 0):<10.4f} "
            f"{r.get('baseline_density', 0):<10.4f} "
            f"{r.get('pos_avg_density', 0):<10.4f}"
        )

    print(f"\nValidated (monotonic trend): {validated}/{len(all_results)}")

    # Fully validated features: consensus + monotonic
    fully_validated = [
        r for r in all_results
        if r.get("is_consensus") and r.get("monotonic_trend")
    ]
    if fully_validated:
        print(f"\nFully validated typing features (consensus + monotonic):")
        for r in fully_validated:
            print(f"  Feature {r['feature_idx']}: {r['label']}")


def load_registry(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    main()
