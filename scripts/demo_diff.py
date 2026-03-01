#!/usr/bin/env python3
"""Run the demo prompt through each recommended feature and print diffs.

For each feature: generates a baseline (no steering) and a steered completion,
then prints a unified diff showing exactly what changed.

Usage (on GPU VM):
    python demo_diff.py
    python demo_diff.py --prompt "Implement a binary search function."
    python demo_diff.py --temperature 0.5 --strength 5.0
"""

import argparse
import difflib
import os
import textwrap

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── SAE wrapper (same as run_discovery.py) ───────────────────────────────

class TopKSAE:
    def __init__(self, checkpoint_path, device="cpu"):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = ckpt["state_dict"]
        cfg = ckpt["config"]
        self.d_model = cfg["d_model"]
        self.d_sae = cfg["d_sae"]
        self.k = cfg["k"]
        self.layer = cfg["layer"]
        self.W_enc = sd["W_enc"].to(device)
        self.b_enc = sd["b_enc"].to(device)
        self.W_dec = sd["W_dec"].to(device)
        self.b_pre = sd["b_pre"].to(device)
        self.device = device


# ── Recommended features from FEATURES_FOUND.md ─────────────────────────

DEMO_FEATURES = [
    {"layer": 18, "feature_idx": 13176, "label": "Type annotations",                  "strength": 3.0},
    {"layer": 18, "feature_idx": 9742,  "label": "Error handling",                     "strength": 3.0},
    {"layer": 18, "feature_idx": 16290, "label": "Recursive patterns",                 "strength": 3.0},
    {"layer": 18, "feature_idx": 9344,  "label": "Verbose comments & documentation",   "strength": 3.0},
    {"layer": 27, "feature_idx": 480,   "label": "Functional style (map/filter/lambda)","strength": 3.0},
]

DEFAULT_PROMPT = "Write a Python function that merges two sorted lists."


# ── Helpers ──────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.3):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def steered_generate(model, tokenizer, sae, feature_idx, strength,
                     prompt, max_new_tokens=200, temperature=0.3):
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] > 1:  # prompt prefill — skip
            return output
        vec = sae.W_dec[feature_idx].to(hidden.device, hidden.dtype)
        hidden = hidden + strength * vec
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle = model.model.layers[sae.layer].register_forward_hook(hook)
    try:
        text = generate(model, tokenizer, prompt, max_new_tokens, temperature)
    finally:
        handle.remove()
    return text


def print_diff(baseline, steered, label):
    base_lines = baseline.splitlines(keepends=True)
    steer_lines = steered.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        base_lines, steer_lines,
        fromfile="baseline", tofile=f"steered [{label}]",
        lineterm="",
    ))
    if not diff:
        print("  (no difference)")
        return
    for line in diff:
        line = line.rstrip("\n")
        if line.startswith("+++") or line.startswith("---"):
            print(f"  {line}")
        elif line.startswith("@@"):
            print(f"  \033[36m{line}\033[0m")
        elif line.startswith("+"):
            print(f"  \033[32m{line}\033[0m")
        elif line.startswith("-"):
            print(f"  \033[31m{line}\033[0m")
        else:
            print(f"  {line}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run demo prompt with each feature and print diffs")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to generate from")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--strength", type=float, default=None,
                        help="Override steering strength for all features (default: per-feature)")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sae-dir", default=os.path.expanduser("~/8b_saes"))
    args = parser.parse_args()

    # Load model
    print("Loading Ministral-8B...")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Ministral-8B-Instruct-2410",
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Ministral-8B-Instruct-2410")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Load SAEs
    sae_l18 = TopKSAE(os.path.join(args.sae_dir, "layer_18_sae_checkpoint.pt"), device=args.device)
    sae_l27 = TopKSAE(os.path.join(args.sae_dir, "layer_27_sae_checkpoint.pt"), device=args.device)
    sae_map = {18: sae_l18, 27: sae_l27}
    print(f"  SAEs loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Header
    print()
    print("=" * 70)
    print(f"  Prompt:      {args.prompt}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens:  {args.max_tokens}")
    print("=" * 70)

    # Baseline
    print("\nGenerating baseline (no steering)...")
    baseline = generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature)

    print("\n--- BASELINE OUTPUT ---")
    print(baseline)
    print()

    # Each feature
    for feat in DEMO_FEATURES:
        layer = feat["layer"]
        idx = feat["feature_idx"]
        label = feat["label"]
        strength = args.strength if args.strength is not None else feat["strength"]
        sae = sae_map[layer]

        print("=" * 70)
        print(f"  Feature:   {label}")
        print(f"  Layer:     {layer}")
        print(f"  Index:     {idx}")
        print(f"  Strength:  +{strength}")
        print("=" * 70)

        steered = steered_generate(
            model, tokenizer, sae, idx, strength,
            args.prompt, args.max_tokens, args.temperature,
        )

        print()
        print_diff(baseline, steered, label)

        print(f"\n--- STEERED OUTPUT [{label}] ---")
        print(steered)
        print()

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
