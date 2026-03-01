#!/usr/bin/env python3
"""Characterize the SAE before committing to feature discovery parameters.

Computes:
  - Dictionary size (d_in, d_sae)
  - Decoder vector norm statistics (min, max, mean, percentiles)
  - Mean L0 sparsity on ~200 diverse prompts
  - Dead feature count (features that never fire)
  - Feature activation distribution (histogram of max activations)

Output: results/sae_profile.json + printed summary table.

Usage:
    python 00_explore_sae.py [--n-prompts 200] [--results-dir results]
"""

import argparse
import json
import os

import torch
import numpy as np
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer


# Diverse prompts covering code, prose, math, and mixed content.
# Kept short to fit in memory; quantity matters more than length here.
DIVERSE_PROMPTS = [
    # Code prompts (various languages and styles)
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "function mergeSort(arr) {\n  if (arr.length <= 1) return arr;\n  const mid = Math.floor(arr.length / 2);",
    "public class LinkedList<T> {\n    private Node<T> head;\n    public void add(T value) {",
    "fn binary_search(arr: &[i32], target: i32) -> Option<usize> {\n    let mut low = 0;",
    "import pandas as pd\ndf = pd.read_csv('data.csv')\ndf.groupby('category').agg({'price': 'mean'})",
    "SELECT u.name, COUNT(o.id) as order_count\nFROM users u\nJOIN orders o ON u.id = o.user_id\nGROUP BY u.name",
    "class Stack:\n    def __init__(self):\n        self._items = []\n    def push(self, item):\n        self._items.append(item)",
    "const express = require('express');\nconst app = express();\napp.get('/api/users', async (req, res) => {",
    "async def fetch_data(url: str) -> dict:\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as resp:",
    "try:\n    result = dangerous_operation()\nexcept ValueError as e:\n    logger.error(f'Validation failed: {e}')\n    raise",
    "interface Config {\n  host: string;\n  port: number;\n  debug?: boolean;\n}\nfunction loadConfig(path: string): Config {",
    "from typing import TypeVar, Generic\nT = TypeVar('T')\nclass TreeNode(Generic[T]):\n    def __init__(self, value: T):",
    "#include <vector>\n#include <algorithm>\nstd::vector<int> merge(std::vector<int>& a, std::vector<int>& b) {",
    "package main\nimport \"fmt\"\nfunc main() {\n    ch := make(chan int)\n    go func() { ch <- 42 }()",
    "import React, { useState, useEffect } from 'react';\nfunction UserList() {\n  const [users, setUsers] = useState([]);",
    "CREATE TABLE products (\n  id SERIAL PRIMARY KEY,\n  name VARCHAR(255) NOT NULL,\n  price DECIMAL(10,2)",
    "#!/bin/bash\nfor file in *.log; do\n  if [ -f \"$file\" ]; then\n    gzip \"$file\"\n  fi\ndone",
    "@app.route('/login', methods=['POST'])\ndef login():\n    username = request.form['username']\n    password = request.form['password']",
    "model = tf.keras.Sequential([\n    tf.keras.layers.Dense(128, activation='relu'),\n    tf.keras.layers.Dropout(0.2),",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]",
    # Prose prompts (various topics)
    "The French Revolution began in 1789 with the storming of the Bastille. The subsequent Reign of Terror saw thousands executed.",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy. Chlorophyll in the leaves absorbs light.",
    "The theory of general relativity, published by Albert Einstein in 1915, describes gravity as the curvature of spacetime.",
    "Mount Everest, standing at 8,849 meters, is the highest peak above sea level. Located in the Himalayas, it straddles Nepal and Tibet.",
    "The human brain contains approximately 86 billion neurons, each connected to thousands of other neurons through synapses.",
    "Jazz music originated in the African-American communities of New Orleans in the late 19th and early 20th centuries.",
    "The Amazon rainforest produces roughly 20 percent of the world's oxygen and houses an estimated 10 percent of all known species.",
    "Democracy, from the Greek word demos meaning people, is a system of government where citizens exercise power through voting.",
    "The Industrial Revolution transformed manufacturing processes in the late 18th century, shifting from hand production to machine-based methods.",
    "Coral reefs are built by colonies of tiny animals called polyps. They support approximately 25 percent of all marine life.",
    "Shakespeare wrote 37 plays and 154 sonnets during his lifetime. His works have been translated into every major language on earth.",
    "The periodic table organizes elements by atomic number and electron configuration. Dmitri Mendeleev published the first version in 1869.",
    "Plate tectonics describes the movement of large sections of Earth's crust. Continental drift was first proposed by Alfred Wegener in 1912.",
    "The Silk Road was an ancient network of trade routes connecting East Asia to the Mediterranean world through Central Asia.",
    "Vaccines work by training the immune system to recognize and combat pathogens. Edward Jenner developed the first vaccine in 1796.",
    "The Great Barrier Reef is the world's largest coral reef system, stretching over 2,300 kilometers along Australia's northeastern coast.",
    "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape once past the event horizon.",
    "The printing press, invented by Johannes Gutenberg around 1440, revolutionized the spread of information across Europe.",
    "Migration patterns of monarch butterflies span thousands of miles, from Canada to central Mexico, guided by an internal magnetic compass.",
    "The Renaissance was a cultural movement that began in Italy during the 14th century, marked by renewed interest in classical learning.",
    # Mixed / instruction-following prompts
    "Write a haiku about the ocean:\nWaves crash on the shore\nSalt spray kisses weathered rocks\nTide pulls back again",
    "Dear hiring manager, I am writing to express my interest in the software engineering position at your company.",
    "Recipe for chocolate chip cookies: Preheat oven to 375 degrees. Mix butter, sugar, and eggs until creamy.",
    "The meeting notes from Tuesday indicate that the project deadline has been moved to next quarter.",
    "According to recent studies, regular exercise can reduce the risk of heart disease by up to 50 percent.",
    "In a galaxy far, far away, there existed a small planet orbiting twin suns. Its inhabitants had never known darkness.",
    "To assemble the bookshelf, first attach panel A to panel B using the provided screws. Ensure alignment before tightening.",
    "The quarterly earnings report shows a 15 percent increase in revenue compared to the same period last year.",
    "Once upon a time, in a kingdom by the sea, there lived a young princess who could speak to animals.",
    "The patient presents with mild fever and persistent cough lasting three days. No history of respiratory conditions.",
]


def compute_l0_sparsity(
    model: HookedTransformer,
    sae: SAE,
    hook_point: str,
    prompts: list[str],
) -> dict:
    """Compute L0 sparsity (average number of active features per token)."""
    all_l0s = []
    all_max_acts = []
    ever_fired = torch.zeros(sae.cfg.d_sae, dtype=torch.bool, device="cpu")

    for prompt in tqdm(prompts, desc="Computing L0 sparsity"):
        tokens = model.to_tokens(prompt, prepend_bos=False)
        if tokens.shape[1] > 256:
            tokens = tokens[:, :256]

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_point)

        acts = cache[hook_point]  # [1, seq_len, d_model]

        # Encode each token position through the SAE
        for pos in range(acts.shape[1]):
            token_acts = acts[0, pos, :].float().unsqueeze(0)  # [1, d_model]
            feat_acts = sae.encode(token_acts).squeeze(0)  # [d_sae]

            active_mask = feat_acts > 0
            l0 = active_mask.sum().item()
            all_l0s.append(l0)

            # Track which features ever fire
            ever_fired |= active_mask.cpu()

            # Track max activation per feature
            if len(all_max_acts) == 0:
                all_max_acts = feat_acts.cpu().clone()
            else:
                all_max_acts = torch.maximum(all_max_acts, feat_acts.cpu())

        del cache
        torch.cuda.empty_cache()

    dead_features = (~ever_fired).sum().item()
    total_features = sae.cfg.d_sae

    return {
        "mean_l0": float(np.mean(all_l0s)),
        "median_l0": float(np.median(all_l0s)),
        "std_l0": float(np.std(all_l0s)),
        "min_l0": int(min(all_l0s)),
        "max_l0": int(max(all_l0s)),
        "total_tokens_analyzed": len(all_l0s),
        "dead_features": int(dead_features),
        "alive_features": int(total_features - dead_features),
        "dead_feature_fraction": round(dead_features / total_features, 4),
        "max_activation_per_feature": {
            "min": float(all_max_acts.min()),
            "max": float(all_max_acts.max()),
            "mean": float(all_max_acts.mean()),
            "median": float(all_max_acts.median()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Characterize SAE before feature discovery")
    parser.add_argument("--n-prompts", type=int, default=200, help="Number of prompts for L0/dead feature analysis")
    parser.add_argument("--results-dir", type=str, default="results", help="Output directory")
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

    # ── Basic SAE dimensions ──
    d_sae = sae.cfg.d_sae
    d_in = sae.cfg.d_in

    print(f"\n{'='*60}")
    print("SAE Profile")
    print(f"{'='*60}")
    print(f"  Dictionary size (d_sae): {d_sae:,}")
    print(f"  Input dimension (d_in):  {d_in}")
    print(f"  Hook point:              {hook_point}")

    # ── Decoder vector norms ──
    print("\nComputing decoder vector norms...")
    with torch.no_grad():
        norms = sae.W_dec.norm(dim=1).cpu().float().numpy()

    norm_stats = {
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
        "mean": float(np.mean(norms)),
        "median": float(np.median(norms)),
        "std": float(np.std(norms)),
        "p5": float(np.percentile(norms, 5)),
        "p25": float(np.percentile(norms, 25)),
        "p75": float(np.percentile(norms, 75)),
        "p95": float(np.percentile(norms, 95)),
    }

    print(f"\n  Decoder Vector Norms:")
    print(f"    Min:    {norm_stats['min']:.4f}")
    print(f"    P5:     {norm_stats['p5']:.4f}")
    print(f"    P25:    {norm_stats['p25']:.4f}")
    print(f"    Median: {norm_stats['median']:.4f}")
    print(f"    Mean:   {norm_stats['mean']:.4f}")
    print(f"    P75:    {norm_stats['p75']:.4f}")
    print(f"    P95:    {norm_stats['p95']:.4f}")
    print(f"    Max:    {norm_stats['max']:.4f}")
    print(f"    Std:    {norm_stats['std']:.4f}")

    # ── Encoder vector norms (for reference) ──
    print("\nComputing encoder vector norms...")
    with torch.no_grad():
        enc_norms = sae.W_enc.norm(dim=0).cpu().float().numpy()

    enc_norm_stats = {
        "min": float(np.min(enc_norms)),
        "max": float(np.max(enc_norms)),
        "mean": float(np.mean(enc_norms)),
        "median": float(np.median(enc_norms)),
        "std": float(np.std(enc_norms)),
    }

    print(f"\n  Encoder Vector Norms:")
    print(f"    Min:    {enc_norm_stats['min']:.4f}")
    print(f"    Median: {enc_norm_stats['median']:.4f}")
    print(f"    Mean:   {enc_norm_stats['mean']:.4f}")
    print(f"    Max:    {enc_norm_stats['max']:.4f}")

    # ── L0 sparsity and dead features ──
    prompts = DIVERSE_PROMPTS[:args.n_prompts]
    print(f"\nComputing L0 sparsity and dead features on {len(prompts)} prompts...")

    sparsity_stats = compute_l0_sparsity(model, sae, hook_point, prompts)

    print(f"\n  L0 Sparsity (active features per token):")
    print(f"    Mean:   {sparsity_stats['mean_l0']:.1f}")
    print(f"    Median: {sparsity_stats['median_l0']:.1f}")
    print(f"    Std:    {sparsity_stats['std_l0']:.1f}")
    print(f"    Range:  [{sparsity_stats['min_l0']}, {sparsity_stats['max_l0']}]")
    print(f"    Total tokens analyzed: {sparsity_stats['total_tokens_analyzed']:,}")

    print(f"\n  Dead Features (never fire on any prompt):")
    print(f"    Dead:  {sparsity_stats['dead_features']:,} / {d_sae:,} ({sparsity_stats['dead_feature_fraction']:.1%})")
    print(f"    Alive: {sparsity_stats['alive_features']:,}")

    # ── Steering strength implications ──
    print(f"\n{'='*60}")
    print("Steering Strength Implications")
    print(f"{'='*60}")
    print(f"  At strength=3.0:")
    print(f"    Min perturbation magnitude: {3.0 * norm_stats['min']:.2f}")
    print(f"    Median perturbation magnitude: {3.0 * norm_stats['median']:.2f}")
    print(f"    Max perturbation magnitude: {3.0 * norm_stats['max']:.2f}")
    print(f"    Ratio max/min: {norm_stats['max'] / norm_stats['min']:.1f}x")
    print(f"\n  Recommendation: normalize steering by decoder norm to make")
    print(f"  strength comparable across features, or use norm-aware scaling.")

    # ── Save profile ──
    profile = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "sae": "tylercosgrove/mistral-7b-sparse-autoencoder-layer16",
        "hook_point": hook_point,
        "d_in": d_in,
        "d_sae": d_sae,
        "decoder_norms": norm_stats,
        "encoder_norms": enc_norm_stats,
        "sparsity": sparsity_stats,
    }

    output_path = os.path.join(args.results_dir, "sae_profile.json")
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"\nSaved SAE profile to {output_path}")


if __name__ == "__main__":
    main()
