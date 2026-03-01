#!/usr/bin/env python3
"""Collect SAE activations via prompt-and-observe for coding properties.

For each property, prompts the model to generate code exhibiting (target) or
lacking (control) the property, then records which SAE features fire most
strongly on the generated output. This is the blog-post method: let the model
produce the code, then observe the internal representations.

Saves per-property JSON files with generated outputs and top-K features.

Usage:
    python 01_collect_activations.py --property error_handling
    python 01_collect_activations.py --property all
    python 01_collect_activations.py --property all --n-prompts 5  # quick test
"""

import argparse
import json
import os
import sys

import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer

# Allow imports from parent/sibling
sys.path.insert(0, os.path.dirname(__file__))
from properties import PROPERTIES


def collect_top_features(
    model: HookedTransformer,
    sae: SAE,
    hook_point: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> dict:
    """Generate text from a prompt and record which SAE features fire.

    This is the core prompt-and-observe step:
    1. Generate a completion from the prompt
    2. Run the full output through the model to get activations
    3. Encode activations with the SAE
    4. Record the top-K firing features (mean-pooled and last-token)
    """
    tokens = model.to_tokens(prompt, prepend_bos=False)

    with torch.no_grad():
        # Step 1: Generate the completion
        output_tokens = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        output_text = model.tokenizer.decode(
            output_tokens[0], skip_special_tokens=True
        )

        # Step 2: Run the full output through the model for activations
        # Truncate to avoid OOM on very long sequences
        if output_tokens.shape[1] > 512:
            output_tokens = output_tokens[:, :512]

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

    # Last-token activations
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
        "num_tokens": int(output_tokens.shape[1]),
        "mean_pooled_features": mean_features,
        "last_token_features": last_features,
    }


def collect_property(
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
    """Run prompt-and-observe collection for a single property."""
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

    # Collect target activations
    print("\nCollecting features from TARGET prompts (should exhibit property)...")
    target_results = []
    for prompt in tqdm(target_prompts, desc="Target"):
        result = collect_top_features(
            model, sae, hook_point, prompt,
            max_new_tokens, temperature, top_k,
        )
        target_results.append(result)

    # Collect control activations
    print("\nCollecting features from CONTROL prompts (should lack property)...")
    control_results = []
    for prompt in tqdm(control_prompts, desc="Control"):
        result = collect_top_features(
            model, sae, hook_point, prompt,
            max_new_tokens, temperature, top_k,
        )
        control_results.append(result)

    return {
        "property": property_name,
        "description": property_config["description"],
        "n_target_prompts": len(target_prompts),
        "n_control_prompts": len(control_prompts),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "target_results": target_results,
        "control_results": control_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prompt-and-observe activation collection for coding properties"
    )
    parser.add_argument(
        "--property", type=str, required=True,
        choices=list(PROPERTIES.keys()) + ["all"],
        help="Which coding property to investigate",
    )
    parser.add_argument("--n-prompts", type=int, default=None,
                        help="Max prompts per category (for quick tests)")
    parser.add_argument("--max-tokens", type=int, default=300,
                        help="Max new tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-K features to record per prompt")
    parser.add_argument("--results-dir", type=str,
                        default="scripts/prompt_observe/results",
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
    sae = sae_tuple[0] if isinstance(sae_tuple, tuple) else sae_tuple
    sae = sae.to(model.cfg.device)

    hook_point = sae.cfg.metadata["hook_name"]
    print(f"  SAE loaded, hook point: {hook_point}")
    print(f"  SAE dimensions: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    # Determine which properties to process
    if args.property == "all":
        properties = list(PROPERTIES.items())
    else:
        properties = [(args.property, PROPERTIES[args.property])]

    for prop_name, prop_config in properties:
        result = collect_property(
            model, sae, hook_point,
            prop_name, prop_config,
            args.max_tokens, args.temperature, args.top_k,
            args.n_prompts,
        )

        # Save per-property results
        output_path = os.path.join(
            args.results_dir, f"activations_{prop_name}.json"
        )
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved {prop_name} activations to {output_path}")

        # Print quick stats
        target_feats = sum(
            len(r["mean_pooled_features"]) for r in result["target_results"]
        )
        control_feats = sum(
            len(r["mean_pooled_features"]) for r in result["control_results"]
        )
        print(f"  Target: {target_feats} total features across {result['n_target_prompts']} prompts")
        print(f"  Control: {control_feats} total features across {result['n_control_prompts']} prompts")

    print("\nCollection complete.")


if __name__ == "__main__":
    main()
