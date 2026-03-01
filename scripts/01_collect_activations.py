#!/usr/bin/env python3
"""Collect SAE activations for code and non-code prompts.

Loads MultiPL-E code prompts across 8 languages (~1,280 prompts) and ~300
filtered Alpaca non-code prompts, runs each through Mistral 7B, encodes the
last-token residual stream activations with the layer 16 SAE, and records the
top-50 firing features per prompt.

Usage:
    python 01_collect_activations.py [--top-k 50] [--max-noncode 300]
"""

import argparse
import json
import os
import re
import time

import torch
from datasets import load_dataset
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer


LANGUAGES = ["py", "js", "java", "cpp", "rs", "ts", "sh", "r"]

# MultiPL-E doesn't have a "humaneval-py" config — Python comes from original HumanEval
MULTIPL_E_LANGUAGES = ["js", "java", "cpp", "rs", "ts", "sh", "r"]


def load_code_prompts() -> list[dict]:
    """Load MultiPL-E prompts across 8 languages as the code dataset.

    Python prompts come from openai_humaneval (the original source).
    Other languages come from nuprl/MultiPL-E transpilations.
    ~160 problems x 8 languages = ~1,280 code prompts.
    """
    prompts = []

    # Python from original HumanEval
    try:
        ds = load_dataset("openai_humaneval", split="test")
        for row in ds:
            prompts.append({
                "text": row["prompt"],
                "source": "humaneval",
                "language": "py",
                "id": f"py_{row['task_id']}",
            })
    except Exception as e:
        print(f"  Warning: Could not load Python HumanEval: {e}")

    # Other languages from MultiPL-E
    for lang in MULTIPL_E_LANGUAGES:
        try:
            ds = load_dataset("nuprl/MultiPL-E", f"humaneval-{lang}", split="test")
            for row in ds:
                prompts.append({
                    "text": row["prompt"],
                    "source": "multipl-e",
                    "language": lang,
                    "id": f"{lang}_{row['name']}",
                })
        except Exception as e:
            print(f"  Warning: Could not load language '{lang}': {e}")

    return prompts


def load_noncode_prompts(max_prompts: int) -> list[dict]:
    """Load filtered Alpaca prompts as the non-code contrast set.

    Filters out any instruction containing code-related keywords, and any
    response containing backticks, indentation blocks, or = assignments.
    """
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    exclude_keywords = re.compile(
        r"\b(code|function|script|program|algorithm|write a|implement|"
        r"python|javascript|html|css|sql|debug|compile|variable|class|"
        r"method|loop|array|string|integer|boolean|syntax|api|http|json|"
        r"xml|database|query|server|terminal|command|shell|bash|linux|"
        r"docker|git|regex|library|framework|module|package|import|"
        r"return|print|output|input|file|directory|path|def |int |"
        r"float |list |dict |tuple |set )\b",
        re.IGNORECASE,
    )
    code_chars = re.compile(r"(`|    \S|={1}\s*[\w\"\'\[\{(]|;$|\{|\})", re.MULTILINE)

    prompts = []
    for row in ds:
        instruction = row["instruction"]
        response = row.get("output", "")

        if exclude_keywords.search(instruction):
            continue
        if exclude_keywords.search(response):
            continue
        if code_chars.search(response):
            continue

        # Use the instruction (+ input if present) as the prompt
        text = instruction
        if row.get("input", "").strip():
            text = f"{instruction}\n{row['input']}"

        prompts.append({
            "text": text,
            "source": "alpaca",
            "id": f"alpaca_{len(prompts)}",
        })

        if len(prompts) >= max_prompts:
            break

    return prompts


def collect_activations(
    model: HookedTransformer,
    sae: SAE,
    hook_point: str,
    prompts: list[dict],
    top_k: int,
    label: str,
) -> list[dict]:
    """Run each prompt through the model and record top-K SAE feature activations."""
    results = []

    for prompt_info in tqdm(prompts, desc=f"Collecting {label} activations"):
        try:
            text = prompt_info["text"]
            tokens = model.to_tokens(text, prepend_bos=False)

            # Truncate to avoid OOM on very long prompts
            if tokens.shape[1] > 512:
                tokens = tokens[:, :512]

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens, names_filter=hook_point
                )

            # Encode all token positions through the SAE and mean-pool
            # in feature space.  Using only the last token discards
            # activations from all other positions.
            acts = cache[hook_point]  # [1, seq_len, d_model]
            all_feat_acts = sae.encode(acts[0].float())  # [seq_len, d_sae]
            feature_acts = all_feat_acts.mean(dim=0)  # [d_sae]

            # Get top-K features by activation value
            top_values, top_indices = torch.topk(feature_acts, k=top_k)

            top_features = [
                {"feature_idx": int(idx), "activation": float(val)}
                for idx, val in zip(top_indices.cpu(), top_values.cpu())
                if float(val) > 0  # Only include actually firing features
            ]

            record = {
                "id": prompt_info["id"],
                "source": prompt_info["source"],
                "text": text[:500],  # Truncate text for storage
                "num_tokens": int(tokens.shape[1]),
                "top_features": top_features,
            }
            if "language" in prompt_info:
                record["language"] = prompt_info["language"]
            results.append(record)

        except Exception as e:
            print(f"Error processing {prompt_info['id']}: {e}")
            record = {
                "id": prompt_info["id"],
                "source": prompt_info["source"],
                "text": prompt_info["text"][:500],
                "error": str(e),
                "top_features": [],
            }
            if "language" in prompt_info:
                record["language"] = prompt_info["language"]
            results.append(record)

        # Free VRAM
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Collect SAE activations")
    parser.add_argument("--top-k", type=int, default=50, help="Number of top features to record per prompt")
    parser.add_argument("--max-noncode", type=int, default=300, help="Max non-code prompts to process")
    parser.add_argument("--results-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Load datasets
    print(f"Loading code prompts (MultiPL-E, {len(LANGUAGES)} languages)...")
    code_prompts = load_code_prompts()
    print(f"  Loaded {len(code_prompts)} code prompts")
    for lang in LANGUAGES:
        count = sum(1 for p in code_prompts if p.get("language") == lang)
        print(f"    {lang}: {count}")

    print(f"Loading non-code prompts (filtered Alpaca, max {args.max_noncode})...")
    noncode_prompts = load_noncode_prompts(args.max_noncode)
    print(f"  Loaded {len(noncode_prompts)} non-code prompts")

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

    # Read hook point from SAE metadata
    hook_point = sae.cfg.metadata["hook_name"]
    print(f"  SAE loaded, hook point: {hook_point}")
    print(f"  SAE dimensions: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    # Collect activations
    print("\n--- Collecting code activations ---")
    code_results = collect_activations(
        model, sae, hook_point, code_prompts, args.top_k, "code"
    )

    print("\n--- Collecting non-code activations ---")
    noncode_results = collect_activations(
        model, sae, hook_point, noncode_prompts, args.top_k, "noncode"
    )

    # Save results
    code_path = os.path.join(args.results_dir, "activations_code.json")
    noncode_path = os.path.join(args.results_dir, "activations_noncode.json")

    with open(code_path, "w") as f:
        json.dump(code_results, f, indent=2)
    print(f"\nSaved {len(code_results)} code activation records to {code_path}")

    with open(noncode_path, "w") as f:
        json.dump(noncode_results, f, indent=2)
    print(f"Saved {len(noncode_results)} non-code activation records to {noncode_path}")

    # Print summary
    code_with_features = sum(1 for r in code_results if r["top_features"])
    noncode_with_features = sum(1 for r in noncode_results if r["top_features"])
    print(f"\nSummary:")
    print(f"  Code prompts with features: {code_with_features}/{len(code_results)}")
    print(f"  Non-code prompts with features: {noncode_with_features}/{len(noncode_results)}")
    print(f"  Per-language breakdown:")
    for lang in LANGUAGES:
        lang_total = sum(1 for r in code_results if r.get("language") == lang)
        lang_ok = sum(1 for r in code_results if r.get("language") == lang and r["top_features"])
        print(f"    {lang}: {lang_ok}/{lang_total}")


if __name__ == "__main__":
    main()
