#!/usr/bin/env python3
"""Collect SAE activations for typed and untyped code prompts.

Loads paired typed/untyped code snippets from the dataset, runs each through
Mistral 7B, encodes residual stream activations with the layer 16 SAE, and
records the top-K firing features per prompt. Collects both last-token and
mean-pooled activations.

Usage:
    python 01_collect_activations.py [--top-k 50] [--results-dir scripts/typing/results]
"""

import argparse
import json
import os

import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer


def load_snippets(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def collect_activations(
    model: HookedTransformer,
    sae: SAE,
    hook_point: str,
    snippets: list[dict],
    top_k: int,
    label: str,
) -> list[dict]:
    """Run each snippet through the model and record top-K SAE feature activations.

    Collects both last-token and mean-pooled activations per prompt.
    """
    results = []

    for snippet in tqdm(snippets, desc=f"Collecting {label} activations"):
        try:
            text = snippet["code"]
            tokens = model.to_tokens(text, prepend_bos=False)

            # Truncate to avoid OOM on very long prompts
            if tokens.shape[1] > 512:
                tokens = tokens[:, :512]

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens, names_filter=hook_point
                )

            acts = cache[hook_point]  # [1, seq_len, d_model]

            # --- Last-token activations ---
            last_token_acts = acts[0, -1, :]  # [d_model]
            last_feat_acts = sae.encode(last_token_acts.float().unsqueeze(0)).squeeze(0)

            top_values, top_indices = torch.topk(last_feat_acts, k=top_k)
            last_token_features = [
                {"feature_idx": int(idx), "activation": float(val)}
                for idx, val in zip(top_indices.cpu(), top_values.cpu())
                if float(val) > 0
            ]

            # --- Mean-pooled activations ---
            mean_acts = acts[0].mean(dim=0)  # [d_model]
            mean_feat_acts = sae.encode(mean_acts.float().unsqueeze(0)).squeeze(0)

            top_values_m, top_indices_m = torch.topk(mean_feat_acts, k=top_k)
            mean_pooled_features = [
                {"feature_idx": int(idx), "activation": float(val)}
                for idx, val in zip(top_indices_m.cpu(), top_values_m.cpu())
                if float(val) > 0
            ]

            record = {
                "pair_id": snippet["pair_id"],
                "category": snippet["category"],
                "language": snippet["language"],
                "language_family": snippet["language_family"],
                "typing": snippet["typing"],
                "text": text[:500],
                "num_tokens": int(tokens.shape[1]),
                "top_features": last_token_features,
                "mean_pooled_features": mean_pooled_features,
            }
            results.append(record)

        except Exception as e:
            print(f"Error processing {snippet['pair_id']}: {e}")
            record = {
                "pair_id": snippet["pair_id"],
                "category": snippet["category"],
                "language": snippet["language"],
                "language_family": snippet["language_family"],
                "typing": snippet["typing"],
                "text": snippet["code"][:500],
                "error": str(e),
                "top_features": [],
                "mean_pooled_features": [],
            }
            results.append(record)

        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Collect SAE activations for typing experiment")
    parser.add_argument("--top-k", type=int, default=50, help="Number of top features to record per prompt")
    parser.add_argument("--dataset-dir", type=str, default="scripts/typing/dataset", help="Dataset directory")
    parser.add_argument("--results-dir", type=str, default="scripts/typing/results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Load dataset
    typed_path = os.path.join(args.dataset_dir, "typed_snippets.json")
    untyped_path = os.path.join(args.dataset_dir, "untyped_snippets.json")

    if not os.path.exists(typed_path) or not os.path.exists(untyped_path):
        print("Error: dataset files not found. Run 00_generate_dataset.py first.")
        return

    print("Loading dataset...")
    typed_snippets = load_snippets(typed_path)
    untyped_snippets = load_snippets(untyped_path)
    print(f"  Typed snippets: {len(typed_snippets)}")
    print(f"  Untyped snippets: {len(untyped_snippets)}")

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
    print(f"  SAE dimensions: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    # Collect activations
    print("\n--- Collecting typed activations ---")
    typed_results = collect_activations(
        model, sae, hook_point, typed_snippets, args.top_k, "typed"
    )

    print("\n--- Collecting untyped activations ---")
    untyped_results = collect_activations(
        model, sae, hook_point, untyped_snippets, args.top_k, "untyped"
    )

    # Save results
    typed_out = os.path.join(args.results_dir, "activations_typed.json")
    untyped_out = os.path.join(args.results_dir, "activations_untyped.json")

    with open(typed_out, "w") as f:
        json.dump(typed_results, f, indent=2)
    print(f"\nSaved {len(typed_results)} typed activation records to {typed_out}")

    with open(untyped_out, "w") as f:
        json.dump(untyped_results, f, indent=2)
    print(f"Saved {len(untyped_results)} untyped activation records to {untyped_out}")

    # Print summary
    typed_ok = sum(1 for r in typed_results if r["top_features"])
    untyped_ok = sum(1 for r in untyped_results if r["top_features"])
    print(f"\nSummary:")
    print(f"  Typed prompts with features: {typed_ok}/{len(typed_results)}")
    print(f"  Untyped prompts with features: {untyped_ok}/{len(untyped_results)}")

    for lang in ["typescript", "python"]:
        t_count = sum(1 for r in typed_results if r["language"] == lang and r["top_features"])
        t_total = sum(1 for r in typed_results if r["language"] == lang)
        u_count = sum(1 for r in untyped_results if r["language"] == lang and r["top_features"])
        u_total = sum(1 for r in untyped_results if r["language"] == lang)
        print(f"  {lang}: typed {t_count}/{t_total}, untyped {u_count}/{u_total}")


if __name__ == "__main__":
    main()
