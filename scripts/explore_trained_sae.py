#!/usr/bin/env python3
"""Explore a trained SAE: profile quality, discover features, test steering.

Usage:
    python explore_trained_sae.py --sae-path ~/checkpoints/code_sae_v1/final
"""

import argparse
import json
import os

import torch
import numpy as np
from tqdm import tqdm
from transformer_lens import HookedTransformer

# ── Diverse test prompts ─────────────────────────────────────────────────

CODE_PROMPTS = {
    "typed_python": [
        "def binary_search(arr: list[int], target: int) -> int:\n    low: int = 0\n    high: int = len(arr) - 1\n    while low <= high:\n        mid: int = (low + high) // 2\n        if arr[mid] == target:\n            return mid",
        "from typing import Optional, Dict\n\ndef get_user(user_id: int) -> Optional[Dict[str, str]]:\n    users: Dict[int, Dict[str, str]] = {}\n    return users.get(user_id)",
        "class Stack:\n    def __init__(self) -> None:\n        self._items: list[int] = []\n    def push(self, item: int) -> None:\n        self._items.append(item)\n    def pop(self) -> int:\n        return self._items.pop()",
    ],
    "untyped_python": [
        "def binary_search(arr, target):\n    low = 0\n    high = len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid",
        "def get_user(user_id):\n    users = {}\n    return users.get(user_id)",
        "class Stack:\n    def __init__(self):\n        self._items = []\n    def push(self, item):\n        self._items.append(item)\n    def pop(self):\n        return self._items.pop()",
    ],
    "error_handling": [
        "try:\n    result = int(user_input)\nexcept ValueError as e:\n    print(f'Invalid input: {e}')\n    result = 0\nexcept TypeError:\n    raise",
        "def safe_divide(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        logger.error('Division by zero')\n        return None\n    finally:\n        cleanup()",
        "try:\n    with open(filepath) as f:\n        data = json.load(f)\nexcept FileNotFoundError:\n    data = {}\nexcept json.JSONDecodeError as e:\n    raise ValueError(f'Corrupt file: {e}')",
    ],
    "no_error_handling": [
        "result = int(user_input)\nprint(result)",
        "def divide(a, b):\n    return a / b",
        "with open(filepath) as f:\n    data = json.load(f)",
    ],
    "functional": [
        "result = list(map(lambda x: x * 2, filter(lambda x: x > 0, numbers)))\ntotal = reduce(lambda a, b: a + b, result)",
        "pipeline = compose(normalize, tokenize, filter_stopwords, stem)\nprocessed = [pipeline(doc) for doc in documents]",
        "from functools import reduce\nword_counts = reduce(lambda acc, w: {**acc, w: acc.get(w, 0) + 1}, words, {})",
    ],
    "imperative": [
        "result = []\nfor x in numbers:\n    if x > 0:\n        result.append(x * 2)\ntotal = 0\nfor r in result:\n    total += r",
        "processed = []\nfor doc in documents:\n    doc = normalize(doc)\n    doc = tokenize(doc)\n    doc = filter_stopwords(doc)\n    processed.append(doc)",
        "word_counts = {}\nfor w in words:\n    if w in word_counts:\n        word_counts[w] += 1\n    else:\n        word_counts[w] = 1",
    ],
    "recursive": [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
        "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result",
        "def tree_depth(node):\n    if node is None:\n        return 0\n    return 1 + max(tree_depth(node.left), tree_depth(node.right))",
    ],
    "iterative": [
        "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        "def flatten(lst):\n    stack = [lst]\n    result = []\n    while stack:\n        current = stack.pop()\n        if isinstance(current, list):\n            stack.extend(reversed(current))\n        else:\n            result.append(current)\n    return result",
        "def tree_depth(root):\n    if root is None:\n        return 0\n    queue = [(root, 1)]\n    max_depth = 0\n    while queue:\n        node, depth = queue.pop(0)\n        max_depth = max(max_depth, depth)\n        if node.left: queue.append((node.left, depth + 1))\n        if node.right: queue.append((node.right, depth + 1))\n    return max_depth",
    ],
    "typescript": [
        "interface User {\n  id: number;\n  name: string;\n  email: string;\n}\n\nfunction getUser(id: number): User | undefined {\n  const users: User[] = [];\n  return users.find(u => u.id === id);\n}",
        "type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };\n\nfunction parseJson<T>(input: string): Result<T, Error> {\n  try {\n    return { ok: true, value: JSON.parse(input) };\n  } catch (e) {\n    return { ok: false, error: e as Error };\n  }\n}",
    ],
    "javascript": [
        "function getUser(id) {\n  const users = [];\n  return users.find(u => u.id === id);\n}",
        "function parseJson(input) {\n  try {\n    return { ok: true, value: JSON.parse(input) };\n  } catch (e) {\n    return { ok: false, error: e };\n  }\n}",
    ],
    "verbose_comments": [
        "# This function performs a binary search on a sorted array.\n# It takes an array and a target value as input.\n# Returns the index of the target if found, otherwise -1.\ndef binary_search(arr, target):\n    # Initialize the low and high pointers\n    low = 0  # Start of the array\n    high = len(arr) - 1  # End of the array\n    # Continue searching while the search space is valid\n    while low <= high:\n        mid = (low + high) // 2  # Calculate the middle index\n        if arr[mid] == target:  # Found the target\n            return mid",
    ],
    "minimal_comments": [
        "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
    ],
    "prose": [
        "The French Revolution began in 1789 with the storming of the Bastille.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "The theory of general relativity describes gravity as spacetime curvature.",
    ],
}

CONTRASTIVE_PAIRS = [
    ("typed_python", "untyped_python", "Type annotations"),
    ("error_handling", "no_error_handling", "Error handling"),
    ("functional", "imperative", "Functional style"),
    ("recursive", "iterative", "Recursion"),
    ("typescript", "javascript", "TypeScript types"),
    ("verbose_comments", "minimal_comments", "Verbose comments"),
]


def load_sae(sae_path, device):
    """Load a trained SAE from a checkpoint directory."""
    from sae_lens import SAE
    # sae_lens converts BatchTopK -> JumpReLU for inference.
    # Use load_from_disk (load_from_pretrained is deprecated).
    sae = SAE.load_from_disk(sae_path, device=device)
    return sae


def collect_activations(model, sae, hook_point, text, device):
    """Get mean-pooled SAE feature activations for a text."""
    tokens = model.to_tokens(text, prepend_bos=True)
    if tokens.shape[1] > 512:
        tokens = tokens[:, :512]

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_point)

    acts = cache[hook_point]  # [1, seq_len, d_model]

    # Encode each token position through the SAE, then mean-pool in feature space.
    # The SAE encoder is nonlinear (top-k), so encode(mean(x)) != mean(encode(x)).
    all_feat_acts = sae.encode(acts[0].to(torch.float32))  # [seq_len, d_sae]
    feat_acts = all_feat_acts.mean(dim=0)  # [d_sae]

    del cache
    torch.cuda.empty_cache()

    return feat_acts.cpu()


def profile_sae(model, sae, hook_point, device):
    """Compute basic SAE quality metrics."""
    print("\n" + "=" * 60)
    print("1. SAE PROFILE")
    print("=" * 60)

    d_sae = sae.cfg.d_sae
    d_in = sae.cfg.d_in
    print(f"  d_in: {d_in}, d_sae: {d_sae} ({d_sae // d_in}x expansion)")
    print(f"  Architecture: {sae.cfg.architecture}")
    print(f"  Hook: {sae.cfg.metadata.get('hook_name', 'unknown')}")
    print(f"  Context size: {sae.cfg.metadata.get('context_size', 'unknown')}")

    # Decoder norms
    with torch.no_grad():
        dec_norms = sae.W_dec.norm(dim=1).cpu().float().numpy()
    print(f"\n  Decoder norms:")
    print(f"    min={dec_norms.min():.4f}  median={np.median(dec_norms):.4f}  "
          f"mean={dec_norms.mean():.4f}  max={dec_norms.max():.4f}")

    # L0 sparsity and dead features on diverse prompts
    all_prompts = []
    for category, prompts in CODE_PROMPTS.items():
        all_prompts.extend(prompts)

    all_l0s = []
    ever_fired = torch.zeros(d_sae, dtype=torch.bool)

    print(f"\n  Computing L0 sparsity on {len(all_prompts)} prompts...")
    for text in tqdm(all_prompts, desc="  L0 scan"):
        tokens = model.to_tokens(text, prepend_bos=True)
        if tokens.shape[1] > 512:
            tokens = tokens[:, :512]

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_point)

        acts = cache[hook_point]
        for pos in range(acts.shape[1]):
            token_acts = acts[0, pos, :].float().unsqueeze(0)
            feat_acts = sae.encode(token_acts).squeeze(0)
            active = feat_acts > 0
            all_l0s.append(active.sum().item())
            ever_fired |= active.cpu()

        del cache
        torch.cuda.empty_cache()

    dead = (~ever_fired).sum().item()
    alive = d_sae - dead
    l0_arr = np.array(all_l0s)

    print(f"\n  L0 sparsity (features per token):")
    print(f"    mean={l0_arr.mean():.1f}  median={np.median(l0_arr):.1f}  "
          f"std={l0_arr.std():.1f}  min={l0_arr.min()}  max={l0_arr.max()}")
    print(f"    L0 varies: {'YES' if l0_arr.std() > 0.5 else 'NO (constant like TopK)'}")

    print(f"\n  Dead features: {dead:,} / {d_sae:,} ({dead/d_sae:.1%})")
    print(f"  Alive features: {alive:,} ({alive/d_sae:.1%})")

    return {
        "d_sae": d_sae,
        "d_in": d_in,
        "dead_features": dead,
        "alive_features": alive,
        "dead_fraction": dead / d_sae,
        "l0_mean": float(l0_arr.mean()),
        "l0_std": float(l0_arr.std()),
        "l0_min": int(l0_arr.min()),
        "l0_max": int(l0_arr.max()),
        "decoder_norm_min": float(dec_norms.min()),
        "decoder_norm_max": float(dec_norms.max()),
        "decoder_norm_mean": float(dec_norms.mean()),
    }


def contrastive_analysis(model, sae, hook_point, device):
    """Find features that differentiate contrastive code property pairs."""
    print("\n" + "=" * 60)
    print("2. CONTRASTIVE FEATURE DISCOVERY")
    print("=" * 60)

    results = {}

    for pos_cat, neg_cat, label in CONTRASTIVE_PAIRS:
        pos_prompts = CODE_PROMPTS[pos_cat]
        neg_prompts = CODE_PROMPTS[neg_cat]

        # Collect activations
        pos_acts = []
        for text in pos_prompts:
            acts = collect_activations(model, sae, hook_point, text, device)
            pos_acts.append(acts)
        pos_acts = torch.stack(pos_acts)  # [n_pos, d_sae]

        neg_acts = []
        for text in neg_prompts:
            acts = collect_activations(model, sae, hook_point, text, device)
            neg_acts.append(acts)
        neg_acts = torch.stack(neg_acts)  # [n_neg, d_sae]

        # Compute differential: mean positive - mean negative
        pos_mean = pos_acts.mean(dim=0)
        neg_mean = neg_acts.mean(dim=0)
        diff = pos_mean - neg_mean

        # Also compute frequency (fraction of prompts where feature fires)
        pos_freq = (pos_acts > 0).float().mean(dim=0)
        neg_freq = (neg_acts > 0).float().mean(dim=0)

        # Score: features that fire more often AND more strongly in positive
        score = diff * (pos_freq - neg_freq + 0.1)  # small bias to not zero out

        # Top features for this property
        top_idx = score.topk(10).indices.tolist()
        top_scores = score.topk(10).values.tolist()

        # Also get top suppressing features (negative direction)
        bot_idx = (-score).topk(5).indices.tolist()
        bot_scores = [score[i].item() for i in bot_idx]

        print(f"\n  --- {label} ({pos_cat} vs {neg_cat}) ---")
        print(f"  Top 10 activating features:")
        for rank, (idx, sc) in enumerate(zip(top_idx, top_scores)):
            pf = pos_freq[idx].item()
            nf = neg_freq[idx].item()
            pa = pos_mean[idx].item()
            na = neg_mean[idx].item()
            print(f"    #{rank+1} feat {idx:5d}  score={sc:+.3f}  "
                  f"pos_freq={pf:.0%} act={pa:.2f}  neg_freq={nf:.0%} act={na:.2f}")

        print(f"  Top 5 suppressing features:")
        for rank, (idx, sc) in enumerate(zip(bot_idx, bot_scores)):
            pf = pos_freq[idx].item()
            nf = neg_freq[idx].item()
            print(f"    #{rank+1} feat {idx:5d}  score={sc:+.3f}  "
                  f"pos_freq={pf:.0%}  neg_freq={nf:.0%}")

        results[label] = {
            "positive_category": pos_cat,
            "negative_category": neg_cat,
            "top_features": [
                {"feature_id": idx, "score": sc,
                 "pos_freq": pos_freq[idx].item(), "neg_freq": neg_freq[idx].item(),
                 "pos_mean_act": pos_mean[idx].item(), "neg_mean_act": neg_mean[idx].item()}
                for idx, sc in zip(top_idx, top_scores)
            ],
            "suppressing_features": [
                {"feature_id": idx, "score": sc}
                for idx, sc in zip(bot_idx, bot_scores)
            ],
        }

    # Check cross-property contamination
    print("\n  --- Cross-property contamination check ---")
    all_top_features = {}
    for label, data in results.items():
        for f in data["top_features"][:5]:
            fid = f["feature_id"]
            all_top_features.setdefault(fid, []).append(label)

    shared = {fid: labels for fid, labels in all_top_features.items() if len(labels) > 1}
    if shared:
        print(f"  WARNING: {len(shared)} features appear in multiple properties:")
        for fid, labels in shared.items():
            print(f"    Feature {fid}: {', '.join(labels)}")
    else:
        print("  No cross-property contamination in top-5 features.")

    return results


def steering_test(model, sae, hook_point, device, contrastive_results):
    """Test whether top features actually steer generation."""
    print("\n" + "=" * 60)
    print("3. STEERING VERIFICATION")
    print("=" * 60)

    test_prompts = [
        "def sort_list(items):\n    ",
        "Implement a function that checks if a string is a palindrome.\n\n",
        "def process_data(data):\n    ",
    ]

    for label, data in contrastive_results.items():
        top_feat = data["top_features"][0]
        feat_id = top_feat["feature_id"]

        print(f"\n  --- {label}: feature {feat_id} (score={top_feat['score']:.3f}) ---")

        steering_vec = sae.W_dec[feat_id].detach().clone()

        for prompt in test_prompts[:2]:
            prompt_short = prompt.strip()[:50]

            # Baseline
            baseline = model.generate(
                prompt, max_new_tokens=150, temperature=0.3, do_sample=True,
            )

            # Steered positive
            def pos_hook(value, hook):
                value[:, :, :] = value + 5.0 * steering_vec.to(value.device, value.dtype)
                return value

            with model.hooks(fwd_hooks=[(hook_point, pos_hook)]):
                steered_pos = model.generate(
                    prompt, max_new_tokens=150, temperature=0.3, do_sample=True,
                )

            # Steered negative
            def neg_hook(value, hook):
                value[:, :, :] = value - 5.0 * steering_vec.to(value.device, value.dtype)
                return value

            with model.hooks(fwd_hooks=[(hook_point, neg_hook)]):
                steered_neg = model.generate(
                    prompt, max_new_tokens=150, temperature=0.3, do_sample=True,
                )

            print(f"\n    Prompt: {prompt_short}...")
            print(f"    BASELINE:\n      {baseline[len(prompt):].strip()[:200]}")
            print(f"    +5.0:\n      {steered_pos[len(prompt):].strip()[:200]}")
            print(f"    -5.0:\n      {steered_neg[len(prompt):].strip()[:200]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae-path", type=str, default=os.path.expanduser("~/checkpoints/code_sae_v1/final"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="sae_exploration_results.json")
    parser.add_argument("--skip-steering", action="store_true", help="Skip generation tests")
    args = parser.parse_args()

    print(f"Loading model on {args.device}...")
    model = HookedTransformer.from_pretrained_no_processing(
        "mistralai/Mistral-7B-Instruct-v0.1",
        device=args.device,
        dtype=torch.float16,
    )
    print(f"  Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print(f"Loading SAE from {args.sae_path}...")
    sae = load_sae(args.sae_path, args.device)
    hook_point = sae.cfg.metadata.get("hook_name", "blocks.16.hook_resid_post")
    print(f"  SAE loaded. Hook: {hook_point}")

    # 1. Profile
    profile = profile_sae(model, sae, hook_point, args.device)

    # 2. Contrastive discovery
    contrastive = contrastive_analysis(model, sae, hook_point, args.device)

    # 3. Steering test
    if not args.skip_steering:
        steering_test(model, sae, hook_point, args.device, contrastive)

    # Save results
    output = {
        "sae_path": args.sae_path,
        "profile": profile,
        "contrastive_features": contrastive,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
