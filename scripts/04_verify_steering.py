#!/usr/bin/env python3
"""Verify steering effect for labeled code features.

For each labeled feature, generates baseline and steered outputs using the
SAE decoder vector as a steering direction. Compares outputs side by side.

Usage:
    python 04_verify_steering.py [--strength 3.0] [--max-tokens 300] [--temperature 0.3]
"""

import argparse
import json
import os

import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer


TEST_PROMPTS = [
    "Write a Python function that",
    "Implement a solution to",
    "Here is a function that calculates",
    "def process_data(",
    "Create a class that handles",
]


def load_registry(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


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
    parser = argparse.ArgumentParser(description="Verify steering effect")
    parser.add_argument("--strength", type=float, default=3.0, help="Steering strength")
    parser.add_argument("--max-tokens", type=int, default=300, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max-features", type=int, default=None, help="Max features to verify")
    parser.add_argument("--prompt-idx", type=int, default=0, help="Index of test prompt to use (0-4)")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory")
    args = parser.parse_args()

    registry_path = os.path.join(args.results_dir, "feature_registry.json")
    if not os.path.exists(registry_path):
        print("Error: feature_registry.json not found. Run 03_label_features.py first.")
        return

    # Load feature registry
    print("Loading feature registry...")
    registry = load_registry(registry_path)
    if args.max_features:
        registry = registry[: args.max_features]
    print(f"  Will verify {len(registry)} features")

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

    # Select test prompt
    test_prompt = TEST_PROMPTS[args.prompt_idx]
    print(f"\nTest prompt: \"{test_prompt}\"")
    print(f"Steering strength: {args.strength}")
    print(f"Temperature: {args.temperature}")

    # Generate baseline once
    print("\nGenerating baseline output...")
    baseline = generate_text(model, test_prompt, args.max_tokens, args.temperature)
    torch.cuda.empty_cache()

    # Verify each feature
    verification_results = []

    for i, feature in enumerate(tqdm(registry, desc="Verifying features")):
        feature_idx = feature["feature_idx"]
        label = feature.get("label", "unknown")

        print(f"\n{'='*80}")
        print(f"Feature {feature_idx}: {label}")
        print(f"{'='*80}")

        try:
            steered = generate_steered(
                model, sae, hook_point, test_prompt,
                feature_idx, args.strength,
                args.max_tokens, args.temperature,
            )

            # Print comparison
            print(f"\n--- BASELINE ---")
            print(baseline[:500])
            print(f"\n--- STEERED (strength={args.strength}) ---")
            print(steered[:500])

            result = {
                "feature_idx": feature_idx,
                "label": label,
                "description": feature.get("description", ""),
                "strength": args.strength,
                "prompt": test_prompt,
                "baseline": baseline,
                "steered": steered,
                "status": "success",
            }

        except Exception as e:
            print(f"  Error: {e}")
            result = {
                "feature_idx": feature_idx,
                "label": label,
                "strength": args.strength,
                "prompt": test_prompt,
                "error": str(e),
                "status": "error",
            }

        verification_results.append(result)
        torch.cuda.empty_cache()

    # Save results
    output_path = os.path.join(args.results_dir, "steering_verification.json")
    with open(output_path, "w") as f:
        json.dump(verification_results, f, indent=2)
    print(f"\n\nSaved {len(verification_results)} verification results to {output_path}")

    # Print summary
    successes = sum(1 for r in verification_results if r["status"] == "success")
    errors = sum(1 for r in verification_results if r["status"] == "error")
    print(f"\nSummary: {successes} successful, {errors} errors out of {len(verification_results)} features")


if __name__ == "__main__":
    main()
