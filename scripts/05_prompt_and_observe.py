#!/usr/bin/env python3
"""Prompt-and-observe rapid feature discovery.

Implements the blog post method: prompt the model to generate code with a
specific target property, record which SAE features fire most strongly,
then optionally cross-reference with differential analysis results.

This is complementary to the contrastive pair approach used in scripts/typing/.
It's faster for initial exploration but less rigorous than paired t-tests.

Usage:
    python 05_prompt_and_observe.py --property "error handling" [--n-prompts 10]
    python 05_prompt_and_observe.py --property "recursion" --cross-ref results/typing_features_ranked.json
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer


# Property definitions: each has prompts that should elicit the property
# and control prompts that request equivalent code WITHOUT the property.
PROPERTY_PROMPTS = {
    "error_handling": {
        "description": "try/catch/except blocks and error handling patterns",
        "target_prompts": [
            "Write a Python function that reads a file and handles all possible errors gracefully with try/except blocks",
            "Write a TypeScript function that fetches data from an API with comprehensive error handling",
            "Write a Python function that parses JSON input with detailed error handling for malformed data",
            "Write a function that connects to a database with retry logic and error handling",
            "Write a robust file upload handler that catches and reports every possible failure mode",
            "Implement a safe division function that handles ZeroDivisionError, TypeError, and returns appropriate error messages",
            "Write a Python function that validates user input with multiple try/except blocks for different error types",
            "Write a TypeScript async function with try/catch that handles network timeouts, 404s, and parse errors",
            "Implement a configuration loader that gracefully handles missing files, invalid YAML, and permission errors",
            "Write a function that processes a CSV file with error handling for encoding issues, missing columns, and type mismatches",
        ],
        "control_prompts": [
            "Write a Python function that reads a file and returns its contents",
            "Write a TypeScript function that fetches data from an API",
            "Write a Python function that parses JSON input",
            "Write a function that connects to a database",
            "Write a file upload handler",
            "Implement a division function",
            "Write a Python function that validates user input",
            "Write a TypeScript async function that makes a network request",
            "Implement a configuration loader",
            "Write a function that processes a CSV file",
        ],
    },
    "verbosity": {
        "description": "Code with extensive comments, docstrings, and documentation",
        "target_prompts": [
            "Write a well-documented Python function with a detailed docstring explaining parameters, return values, examples, and edge cases",
            "Write a heavily commented sorting algorithm where every step is explained",
            "Write a Python class with comprehensive docstrings on every method, including type information and usage examples",
            "Write a function with inline comments explaining the algorithm's time and space complexity at each step",
            "Write a thoroughly documented API endpoint with docstring covering request format, response format, errors, and examples",
            "Write a Python module with a module-level docstring, class docstrings, and method docstrings following Google style",
            "Write a binary search with a comment on every line explaining what it does and why",
            "Write a data processing function with verbose logging and comments explaining each transformation",
            "Write a configuration parser with extensive inline documentation of every option and its default value",
            "Write a well-documented linked list implementation with docstrings explaining invariants and complexity",
        ],
        "control_prompts": [
            "Write a Python function concisely with no comments",
            "Write a sorting algorithm with no comments",
            "Write a Python class with no docstrings",
            "Write a function with no comments",
            "Write an API endpoint with no documentation",
            "Write a Python module with no docstrings",
            "Write a binary search implementation",
            "Write a data processing function",
            "Write a configuration parser",
            "Write a linked list implementation",
        ],
    },
    "functional_style": {
        "description": "Functional programming patterns: map/filter/reduce, immutability, pure functions",
        "target_prompts": [
            "Write Python code using map, filter, and reduce to process a list of numbers",
            "Write a data pipeline using only pure functions and function composition, no mutation",
            "Write TypeScript code using Array.map, filter, and reduce to transform an array of objects",
            "Write a Python function using functools.reduce and lambda to aggregate data without any for loops",
            "Implement a data transformation pipeline using only higher-order functions, no loops or mutation",
            "Write Python code that uses list comprehensions and generator expressions instead of for loops with append",
            "Write a TypeScript solution using method chaining with map/filter/reduce, no temporary variables",
            "Implement a functional approach to tree traversal using recursion and immutable data structures",
            "Write a Python program using itertools functions (chain, groupby, starmap) instead of nested loops",
            "Write a data validation pipeline using function composition where each validator is a pure function",
        ],
        "control_prompts": [
            "Write Python code using for loops to process a list of numbers",
            "Write a data pipeline using classes with mutable state",
            "Write TypeScript code using for loops to transform an array of objects",
            "Write a Python function using a for loop to aggregate data",
            "Implement a data transformation using for loops and temporary variables",
            "Write Python code using for loops with append to build lists",
            "Write a TypeScript solution using for loops and temporary variables",
            "Implement an iterative approach to tree traversal using a stack",
            "Write a Python program using nested for loops",
            "Write a data validation function using if/else chains",
        ],
    },
    "recursion": {
        "description": "Recursive algorithms and recursive data structures",
        "target_prompts": [
            "Write a recursive Python function to compute the nth Fibonacci number",
            "Implement a recursive binary tree traversal (inorder, preorder, postorder)",
            "Write a recursive function to solve the Tower of Hanoi problem",
            "Implement recursive merge sort in Python",
            "Write a recursive function to generate all permutations of a string",
            "Implement a recursive descent parser for arithmetic expressions",
            "Write a recursive function to flatten a deeply nested list",
            "Implement recursive depth-first search on a graph",
            "Write a recursive function to compute the power set of a set",
            "Implement a recursive solution to the N-Queens problem",
        ],
        "control_prompts": [
            "Write an iterative Python function to compute the nth Fibonacci number",
            "Implement an iterative binary tree traversal using a stack",
            "Write an iterative solution to move disks between pegs",
            "Implement iterative merge sort in Python",
            "Write an iterative function to generate all permutations of a string",
            "Implement a parser for arithmetic expressions using a loop and stack",
            "Write an iterative function to flatten a deeply nested list",
            "Implement iterative breadth-first search on a graph",
            "Write an iterative function to compute the power set of a set",
            "Implement an iterative backtracking solution to the N-Queens problem",
        ],
    },
}


def collect_top_features(
    model: HookedTransformer,
    sae: SAE,
    hook_point: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> dict:
    """Generate text and record which SAE features fire most strongly."""
    tokens = model.to_tokens(prompt, prepend_bos=False)

    with torch.no_grad():
        # Generate the completion
        output_tokens = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        output_text = model.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # Run the full output through the model to get activations
        _, cache = model.run_with_cache(output_tokens, names_filter=hook_point)

    acts = cache[hook_point]  # [1, seq_len, d_model]

    # Mean-pooled activations across all positions
    mean_acts = acts[0].mean(dim=0).float().unsqueeze(0)  # [1, d_model]
    mean_feat_acts = sae.encode(mean_acts).squeeze(0)  # [d_sae]

    top_values, top_indices = torch.topk(mean_feat_acts, k=top_k)
    mean_features = [
        {"feature_idx": int(idx), "activation": float(val)}
        for idx, val in zip(top_indices.cpu(), top_values.cpu())
        if float(val) > 0
    ]

    # Last-token activations (for comparison)
    last_acts = acts[0, -1, :].float().unsqueeze(0)  # [1, d_model]
    last_feat_acts = sae.encode(last_acts).squeeze(0)

    top_values_l, top_indices_l = torch.topk(last_feat_acts, k=top_k)
    last_features = [
        {"feature_idx": int(idx), "activation": float(val)}
        for idx, val in zip(top_indices_l.cpu(), top_values_l.cpu())
        if float(val) > 0
    ]

    del cache
    torch.cuda.empty_cache()

    return {
        "prompt": prompt,
        "output": output_text,
        "mean_pooled_features": mean_features,
        "last_token_features": last_features,
    }


def analyze_property(
    model: HookedTransformer,
    sae: SAE,
    hook_point: str,
    property_name: str,
    property_config: dict,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    n_prompts: int | None,
) -> dict:
    """Run prompt-and-observe analysis for a single property."""
    target_prompts = property_config["target_prompts"]
    control_prompts = property_config["control_prompts"]

    if n_prompts:
        target_prompts = target_prompts[:n_prompts]
        control_prompts = control_prompts[:n_prompts]

    print(f"\n{'='*70}")
    print(f"Property: {property_name}")
    print(f"Description: {property_config['description']}")
    print(f"Target prompts: {len(target_prompts)}, Control prompts: {len(control_prompts)}")
    print(f"{'='*70}")

    # Collect features from target prompts
    print("\nCollecting features from TARGET prompts...")
    target_results = []
    target_feature_counts = Counter()
    target_feature_activations = defaultdict(list)

    for prompt in tqdm(target_prompts, desc="Target"):
        result = collect_top_features(
            model, sae, hook_point, prompt, max_new_tokens, temperature, top_k
        )
        target_results.append(result)
        for feat in result["mean_pooled_features"]:
            target_feature_counts[feat["feature_idx"]] += 1
            target_feature_activations[feat["feature_idx"]].append(feat["activation"])

    # Collect features from control prompts
    print("\nCollecting features from CONTROL prompts...")
    control_results = []
    control_feature_counts = Counter()
    control_feature_activations = defaultdict(list)

    for prompt in tqdm(control_prompts, desc="Control"):
        result = collect_top_features(
            model, sae, hook_point, prompt, max_new_tokens, temperature, top_k
        )
        control_results.append(result)
        for feat in result["mean_pooled_features"]:
            control_feature_counts[feat["feature_idx"]] += 1
            control_feature_activations[feat["feature_idx"]].append(feat["activation"])

    # Find features that fire significantly more in target than control
    all_features = set(target_feature_counts.keys()) | set(control_feature_counts.keys())
    n_target = len(target_prompts)
    n_control = len(control_prompts)

    differential_features = []
    for feat_idx in all_features:
        target_freq = target_feature_counts[feat_idx] / n_target
        control_freq = control_feature_counts[feat_idx] / n_control
        target_mean_act = (
            sum(target_feature_activations[feat_idx]) / len(target_feature_activations[feat_idx])
            if target_feature_activations[feat_idx] else 0
        )
        control_mean_act = (
            sum(control_feature_activations[feat_idx]) / len(control_feature_activations[feat_idx])
            if control_feature_activations[feat_idx] else 0
        )

        # Differential score: target signal minus control signal
        target_signal = target_freq * target_mean_act
        control_signal = control_freq * control_mean_act
        diff_score = target_signal - control_signal

        if diff_score > 0 and target_freq >= 0.2:  # Fire in at least 20% of target prompts
            differential_features.append({
                "feature_idx": int(feat_idx),
                "differential_score": round(diff_score, 4),
                "target_frequency": round(target_freq, 4),
                "target_mean_activation": round(target_mean_act, 4),
                "control_frequency": round(control_freq, 4),
                "control_mean_activation": round(control_mean_act, 4),
                "specificity_ratio": round(
                    target_signal / (control_signal + 1e-8), 2
                ),
            })

    differential_features.sort(key=lambda x: x["differential_score"], reverse=True)

    # Print top features
    print(f"\nTop 20 differential features for '{property_name}':")
    print(f"{'Rank':<6} {'Feature':<10} {'DiffScore':<12} {'TgtFreq':<10} {'TgtAct':<10} {'CtlFreq':<10} {'CtlAct':<10} {'Ratio':<8}")
    print("-" * 76)
    for i, feat in enumerate(differential_features[:20], 1):
        print(
            f"{i:<6} "
            f"{feat['feature_idx']:<10} "
            f"{feat['differential_score']:<12.4f} "
            f"{feat['target_frequency']:<10.4f} "
            f"{feat['target_mean_activation']:<10.4f} "
            f"{feat['control_frequency']:<10.4f} "
            f"{feat['control_mean_activation']:<10.4f} "
            f"{feat['specificity_ratio']:<8.2f}"
        )

    return {
        "property": property_name,
        "description": property_config["description"],
        "n_target_prompts": n_target,
        "n_control_prompts": n_control,
        "top_differential_features": differential_features[:50],
        "target_results": [
            {"prompt": r["prompt"], "output": r["output"][:500]}
            for r in target_results
        ],
        "control_results": [
            {"prompt": r["prompt"], "output": r["output"][:500]}
            for r in control_results
        ],
    }


def cross_reference(
    property_results: dict,
    cross_ref_path: str,
) -> list[dict]:
    """Cross-reference discovered features with existing differential analysis."""
    with open(cross_ref_path) as f:
        existing = json.load(f)

    # Collect feature indices from the existing analysis
    existing_features = set()
    for key in ["mode1_combined", "mode2_ts_js", "mode2_python", "mode3_paired"]:
        for feat in existing.get(key, []):
            existing_features.add(feat.get("feature_idx"))

    consensus = existing.get("consensus_feature_indices", [])

    overlaps = []
    for feat in property_results["top_differential_features"]:
        feat_idx = feat["feature_idx"]
        if feat_idx in existing_features:
            overlaps.append({
                **feat,
                "in_existing_analysis": True,
                "is_consensus": feat_idx in consensus,
            })

    print(f"\nCross-reference with {cross_ref_path}:")
    print(f"  Features in existing analysis: {len(overlaps)}/{len(property_results['top_differential_features'])}")
    if overlaps:
        for o in overlaps:
            tag = " [CONSENSUS]" if o.get("is_consensus") else ""
            print(f"    Feature {o['feature_idx']}: diff_score={o['differential_score']:.4f}{tag}")

    return overlaps


def main():
    parser = argparse.ArgumentParser(description="Prompt-and-observe feature discovery")
    parser.add_argument("--property", type=str, required=True,
                        choices=list(PROPERTY_PROMPTS.keys()) + ["all"],
                        help="Which code property to investigate")
    parser.add_argument("--n-prompts", type=int, default=None,
                        help="Max prompts per category (target/control)")
    parser.add_argument("--max-tokens", type=int, default=300,
                        help="Max new tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-K features to record per prompt")
    parser.add_argument("--cross-ref", type=str, default=None,
                        help="Path to existing analysis for cross-referencing")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

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
    sae = sae_tuple[0]
    sae = sae.to(model.cfg.device)

    hook_point = sae.cfg.metadata["hook_name"]
    print(f"  SAE loaded, hook point: {hook_point}")
    print(f"  SAE dimensions: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    # Determine which properties to analyze
    if args.property == "all":
        properties = list(PROPERTY_PROMPTS.items())
    else:
        properties = [(args.property, PROPERTY_PROMPTS[args.property])]

    all_results = {}
    for prop_name, prop_config in properties:
        result = analyze_property(
            model, sae, hook_point,
            prop_name, prop_config,
            args.max_tokens, args.temperature, args.top_k,
            args.n_prompts,
        )

        if args.cross_ref and os.path.exists(args.cross_ref):
            result["cross_reference"] = cross_reference(result, args.cross_ref)

        all_results[prop_name] = result

    # Save results
    output_path = os.path.join(
        args.results_dir,
        f"prompt_observe_{args.property}.json"
    )
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {output_path}")

    # Print summary across all properties
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("Summary Across Properties")
        print(f"{'='*70}")
        for prop_name, result in all_results.items():
            top3 = result["top_differential_features"][:3]
            top3_str = ", ".join(
                f"{f['feature_idx']}({f['differential_score']:.3f})"
                for f in top3
            )
            print(f"  {prop_name}: top features = [{top3_str}]")


if __name__ == "__main__":
    main()
