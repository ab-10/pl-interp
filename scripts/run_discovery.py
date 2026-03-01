#!/usr/bin/env python3
"""Quick feature discovery experiment for hackathon demo.

Loads Ministral-8B + custom SAEs (layers 18, 27), generates target/control
completions for 5 code properties, ranks features by differential activation,
verifies steering, and outputs demo_features.json.

Usage (on GPU VM):
    python run_discovery.py [--device cuda:0]
"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── SAE wrapper ──────────────────────────────────────────────────────────

class TopKSAE:
    """Minimal wrapper for custom BatchTopK SAE checkpoints."""

    def __init__(self, checkpoint_path, device="cpu"):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = ckpt["state_dict"]
        cfg = ckpt["config"]

        self.d_model = cfg["d_model"]
        self.d_sae = cfg["d_sae"]
        self.k = cfg["k"]
        self.layer = cfg["layer"]

        self.W_enc = sd["W_enc"].to(device)  # [d_model, d_sae]
        self.b_enc = sd["b_enc"].to(device)  # [d_sae]
        self.W_dec = sd["W_dec"].to(device)  # [d_sae, d_model]
        self.b_pre = sd["b_pre"].to(device)  # [d_model]
        self.device = device

    @torch.no_grad()
    def encode(self, x):
        """Encode activations to sparse feature space. x: [..., d_model] -> [..., d_sae]"""
        x = x.to(self.device).float()
        # Layer norm
        mu = x.mean(dim=-1, keepdim=True)
        x_centered = x - mu
        std = x_centered.std(dim=-1, keepdim=True)
        x_normed = x_centered / (std + 1e-5)
        # Encode
        pre_acts = (x_normed - self.b_pre) @ self.W_enc + self.b_enc
        # Top-k
        topk = torch.topk(pre_acts, k=self.k, dim=-1, sorted=False)
        result = torch.zeros_like(pre_acts)
        result.scatter_(-1, topk.indices, F.relu(topk.values))
        return result

    def to(self, device):
        self.W_enc = self.W_enc.to(device)
        self.b_enc = self.b_enc.to(device)
        self.W_dec = self.W_dec.to(device)
        self.b_pre = self.b_pre.to(device)
        self.device = device
        return self


# ── Prompt definitions ───────────────────────────────────────────────────

PROPERTIES = {
    "type_annotations": {
        "label": "Type annotations",
        "target": [
            "Write a Python function with full type annotations that merges two sorted lists",
            "Write a TypeScript function with explicit types that parses a CSV string into rows",
            "Write a Python function with type hints for parameters and return value that filters a dictionary by value",
            "Write a typed Python function that converts a nested JSON object to a flat dictionary",
            "Write a TypeScript function with generic type parameters that removes duplicates from an array",
        ],
        "control": [
            "Write a Python function that merges two sorted lists",
            "Write a function that parses a CSV string into rows",
            "Write a Python function that filters a dictionary by value",
            "Write a Python function that converts a nested JSON object to a flat dictionary",
            "Write a function that removes duplicates from an array",
        ],
    },
    "error_handling": {
        "label": "Error handling",
        "target": [
            "Write a Python function that reads a JSON file with try/except for FileNotFoundError and JSONDecodeError",
            "Write a function that connects to an API with error handling for timeouts and HTTP errors",
            "Write a Python function with comprehensive error handling that parses user input as a number",
            "Write a function that opens a database connection with try/except and retry logic",
            "Write a function with error handling that writes data to a CSV file",
        ],
        "control": [
            "Write a Python function that reads a JSON file",
            "Write a function that connects to an API",
            "Write a Python function that parses user input as a number",
            "Write a function that opens a database connection",
            "Write a function that writes data to a CSV file",
        ],
    },
    "recursion": {
        "label": "Recursive patterns",
        "target": [
            "Write a recursive Python function that flattens a nested list",
            "Write a recursive function that computes the nth Fibonacci number",
            "Write a recursive function to find all permutations of a string",
            "Write a recursive depth-first search on an adjacency list graph",
            "Write a recursive function that computes the power set of a list",
        ],
        "control": [
            "Write a Python function that flattens a nested list",
            "Write a function that computes the nth Fibonacci number using a loop",
            "Write a function to find all permutations of a string",
            "Write a breadth-first search on an adjacency list graph",
            "Write a function that computes the power set of a list",
        ],
    },
    "verbose_comments": {
        "label": "Verbose comments & documentation",
        "target": [
            "Write a well-documented Python function with a detailed docstring and inline comments that implements binary search",
            "Write a sorting function with a comment on every line explaining the logic",
            "Write a Python class with comprehensive docstrings on every method that implements a stack",
            "Write a heavily commented function that validates an email address",
            "Write a function with detailed inline documentation that parses command-line arguments",
        ],
        "control": [
            "Write a Python binary search function with no comments",
            "Write a sorting function",
            "Write a Python stack class",
            "Write a function that validates an email address",
            "Write a function that parses command-line arguments",
        ],
    },
    "functional_style": {
        "label": "Functional style (map/filter/lambda)",
        "target": [
            "Write Python code using map, filter, and reduce to process a list of numbers",
            "Write a data pipeline using only pure functions and function composition",
            "Write TypeScript using Array.map and filter to transform objects",
            "Write Python using list comprehensions and generator expressions instead of loops",
            "Write a validation pipeline using higher-order functions",
        ],
        "control": [
            "Write Python code using for loops to process a list of numbers",
            "Write a data pipeline using a class with mutable state",
            "Write TypeScript using for loops to transform objects",
            "Write Python using for loops with append",
            "Write a validation function using if/else chains",
        ],
    },
}

NEUTRAL_PROMPTS = [
    "Write a Python function that merges two sorted lists.",
    "Implement a function that checks if a string is a palindrome.",
    "Write a function that counts word frequencies in a string.",
]


# ── Helpers ──────────────────────────────────────────────────────────────

def generate_text(model, tokenizer, prompt, max_new_tokens=300, temperature=0.3):
    """Generate a completion and return (full_ids, prompt_len, output_text)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_ids, prompt_len, text


def capture_activations(model, input_ids, layer_indices):
    """Forward pass capturing residual stream at specified layers.
    Returns dict: layer_idx -> tensor [1, seq_len, d_model]
    """
    captured = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach()
        return hook

    handles = []
    for idx in layer_indices:
        h = model.model.layers[idx].register_forward_hook(make_hook(idx))
        handles.append(h)

    with torch.no_grad():
        model(input_ids.to(model.device))

    for h in handles:
        h.remove()

    return captured


def collect_features_for_prompt(model, tokenizer, saes, prompt, max_new_tokens=300, temperature=0.3):
    """Generate, capture activations, encode per-token through SAEs.
    Returns dict: layer_idx -> {freq: [d_sae], mean_act: [d_sae], text: str}
    """
    output_ids, prompt_len, text = generate_text(
        model, tokenizer, prompt, max_new_tokens, temperature
    )

    layer_indices = [sae.layer for sae in saes]
    captured = capture_activations(model, output_ids, layer_indices)

    result = {}
    for sae in saes:
        acts = captured[sae.layer]  # [1, seq_len, d_model]
        gen_acts = acts[0, prompt_len:]  # [gen_len, d_model] — generation only

        if gen_acts.shape[0] == 0:
            result[sae.layer] = {
                "freq": torch.zeros(sae.d_sae),
                "mean_act": torch.zeros(sae.d_sae),
            }
            continue

        feats = sae.encode(gen_acts)  # [gen_len, d_sae]
        firing = (feats > 0).float()
        freq = firing.mean(dim=0).cpu()  # [d_sae]
        mean_act = (feats.sum(dim=0) / (firing.sum(dim=0) + 1e-8)).cpu()  # [d_sae]

        result[sae.layer] = {"freq": freq, "mean_act": mean_act}

    result["text"] = text
    return result


# ── Phase 1: Generate & Collect ─────────────────────────────────────────

def phase1_generate(model, tokenizer, saes):
    """Generate target/control completions, collect per-token features."""
    print("\n" + "=" * 70)
    print("PHASE 1: Generate & Collect Activations")
    print("=" * 70)

    all_data = {}

    for prop_name, prop_cfg in PROPERTIES.items():
        print(f"\n--- {prop_cfg['label']} ---")
        prop_data = {"target": [], "control": []}

        for group in ["target", "control"]:
            prompts = prop_cfg[group]
            for i, prompt in enumerate(prompts):
                print(f"  {group}[{i}]: {prompt[:60]}...")
                result = collect_features_for_prompt(
                    model, tokenizer, saes, prompt
                )
                prop_data[group].append(result)
                torch.cuda.empty_cache()

        all_data[prop_name] = prop_data

    return all_data


# ── Phase 2: Rank ───────────────────────────────────────────────────────

def phase2_rank(all_data, saes, top_n=5):
    """Differential ranking: target signal minus control signal."""
    print("\n" + "=" * 70)
    print("PHASE 2: Differential Ranking")
    print("=" * 70)

    candidates = {}  # prop_name -> layer -> list of {feature_idx, diff_score, ...}

    for prop_name, prop_data in all_data.items():
        print(f"\n--- {PROPERTIES[prop_name]['label']} ---")
        candidates[prop_name] = {}

        for sae in saes:
            layer = sae.layer

            # Aggregate across prompts
            t_freqs = torch.stack([r[layer]["freq"] for r in prop_data["target"]])
            t_acts = torch.stack([r[layer]["mean_act"] for r in prop_data["target"]])
            c_freqs = torch.stack([r[layer]["freq"] for r in prop_data["control"]])
            c_acts = torch.stack([r[layer]["mean_act"] for r in prop_data["control"]])

            target_mean_freq = t_freqs.mean(dim=0)
            target_mean_act = t_acts.mean(dim=0)
            control_mean_freq = c_freqs.mean(dim=0)
            control_mean_act = c_acts.mean(dim=0)

            # Cross-prompt frequency: in how many target prompts does it fire?
            target_cross_freq = (t_freqs > 0).float().mean(dim=0)

            # Differential score
            target_signal = target_mean_freq * target_mean_act
            control_signal = control_mean_freq * control_mean_act
            diff = target_signal - control_signal

            # Filter
            mask = (diff > 0) & (target_cross_freq >= 0.4)
            valid_indices = mask.nonzero(as_tuple=True)[0]

            if len(valid_indices) == 0:
                print(f"  Layer {layer}: no features passed filter")
                candidates[prop_name][layer] = []
                continue

            valid_diff = diff[valid_indices]
            _, sorted_order = valid_diff.sort(descending=True)
            top_indices = valid_indices[sorted_order[:top_n]]

            layer_candidates = []
            for idx in top_indices:
                idx = idx.item()
                specificity = (target_signal[idx] / (control_signal[idx] + 1e-8)).item()
                layer_candidates.append({
                    "feature_idx": idx,
                    "diff_score": diff[idx].item(),
                    "target_freq": target_mean_freq[idx].item(),
                    "target_act": target_mean_act[idx].item(),
                    "control_freq": control_mean_freq[idx].item(),
                    "control_act": control_mean_act[idx].item(),
                    "cross_prompt_freq": target_cross_freq[idx].item(),
                    "specificity": specificity,
                })

            candidates[prop_name][layer] = layer_candidates

            print(f"  Layer {layer}: top {len(layer_candidates)} features")
            for rank, c in enumerate(layer_candidates):
                print(f"    #{rank+1} feat {c['feature_idx']:5d}  "
                      f"diff={c['diff_score']:.4f}  "
                      f"t_freq={c['target_freq']:.3f}  "
                      f"t_act={c['target_act']:.2f}  "
                      f"c_freq={c['control_freq']:.3f}  "
                      f"spec={c['specificity']:.1f}x")

    return candidates


# ── Phase 3: Verify ─────────────────────────────────────────────────────

def steering_generate(model, tokenizer, sae, feature_idx, strength, prompt,
                      max_new_tokens=200, temperature=0.3):
    """Generate with a single feature steered at the SAE's layer."""

    def steering_hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] > 1:  # prompt prefill — skip
            return output
        vec = sae.W_dec[feature_idx].to(hidden.device, hidden.dtype)
        hidden = hidden + strength * vec
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle = model.model.layers[sae.layer].register_forward_hook(steering_hook)
    try:
        _, _, text = generate_text(model, tokenizer, prompt, max_new_tokens, temperature)
    finally:
        handle.remove()
    return text


def phase3_verify(model, tokenizer, saes, candidates):
    """Steer on neutral prompts, check for visible effect."""
    print("\n" + "=" * 70)
    print("PHASE 3: Steering Verification")
    print("=" * 70)

    sae_map = {sae.layer: sae for sae in saes}
    test_strengths = [3.0, 5.0, 8.0]
    verified = []

    for prop_name, layers_data in candidates.items():
        prop_label = PROPERTIES[prop_name]["label"]
        print(f"\n--- {prop_label} ---")

        for layer, feats in layers_data.items():
            sae = sae_map[layer]
            if not feats:
                continue

            # Test top 3 candidates per layer
            for feat_info in feats[:3]:
                feat_idx = feat_info["feature_idx"]
                print(f"\n  Layer {layer}, feature {feat_idx} (diff={feat_info['diff_score']:.4f})")

                # Generate baselines
                baselines = {}
                for neutral_prompt in NEUTRAL_PROMPTS:
                    short = neutral_prompt[:40]
                    _, _, baseline_text = generate_text(
                        model, tokenizer, neutral_prompt, 200, 0.3
                    )
                    baselines[short] = baseline_text

                best_strength = None
                pass_count = 0

                for strength in test_strengths:
                    strength_ok = 0
                    for neutral_prompt in NEUTRAL_PROMPTS:
                        short = neutral_prompt[:40]
                        steered = steering_generate(
                            model, tokenizer, sae, feat_idx, strength,
                            neutral_prompt, 200, 0.3
                        )

                        # Simple length/content diff as a signal
                        baseline_text = baselines[short]
                        diff_chars = abs(len(steered) - len(baseline_text))
                        same = steered.strip() == baseline_text.strip()

                        if not same and diff_chars > 20:
                            strength_ok += 1

                        print(f"    strength={strength:.1f} prompt='{short}...' "
                              f"baseline_len={len(baseline_text)} steered_len={len(steered)} "
                              f"{'SAME' if same else 'DIFF'}")

                    if strength_ok >= 2 and best_strength is None:
                        best_strength = strength

                    torch.cuda.empty_cache()

                # Also test negative
                neg_ok = 0
                if best_strength:
                    for neutral_prompt in NEUTRAL_PROMPTS:
                        short = neutral_prompt[:40]
                        steered_neg = steering_generate(
                            model, tokenizer, sae, feat_idx, -best_strength,
                            neutral_prompt, 200, 0.3
                        )
                        baseline_text = baselines[short]
                        same = steered_neg.strip() == baseline_text.strip()
                        if not same:
                            neg_ok += 1

                if best_strength:
                    print(f"    -> PASS: demo_strength={best_strength}, neg_ok={neg_ok}/3")
                    verified.append({
                        "layer": layer,
                        "feature_idx": feat_idx,
                        "label": prop_label,
                        "property": prop_name,
                        "demo_strength": best_strength,
                        "neg_strength": -best_strength,
                        "diff_score": feat_info["diff_score"],
                        "specificity": feat_info["specificity"],
                        "verified_on": [p[:40] for p in NEUTRAL_PROMPTS],
                    })
                else:
                    print(f"    -> FAIL: no strength produced visible diff on >=2 prompts")

    return verified


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--sae-dir", default=os.path.expanduser("~/8b_saes"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    # Load model
    print("Loading Ministral-8B...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Ministral-8B-Instruct-2410",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Ministral-8B-Instruct-2410")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Model loaded in {time.time()-t0:.1f}s. "
          f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Load SAEs
    print("Loading SAEs...")
    sae_l18 = TopKSAE(
        os.path.join(args.sae_dir, "layer_18_sae_checkpoint.pt"), device=device
    )
    sae_l27 = TopKSAE(
        os.path.join(args.sae_dir, "layer_27_sae_checkpoint.pt"), device=device
    )
    saes = [sae_l18, sae_l27]
    print(f"  L18: d_sae={sae_l18.d_sae}, k={sae_l18.k}")
    print(f"  L27: d_sae={sae_l27.d_sae}, k={sae_l27.k}")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Phase 1
    t1 = time.time()
    all_data = phase1_generate(model, tokenizer, saes)
    print(f"\nPhase 1 complete in {time.time()-t1:.1f}s")

    # Save intermediate
    # (can't easily serialize tensors to JSON, save ranking data instead)

    # Phase 2
    t2 = time.time()
    candidates = phase2_rank(all_data, saes)
    print(f"\nPhase 2 complete in {time.time()-t2:.1f}s")

    # Save intermediate ranking
    ranking_path = os.path.join(args.output_dir, "ranking.json")
    with open(ranking_path, "w") as f:
        json.dump(candidates, f, indent=2, default=str)
    print(f"Ranking saved to {ranking_path}")

    # Phase 3
    t3 = time.time()
    verified = phase3_verify(model, tokenizer, saes, candidates)
    print(f"\nPhase 3 complete in {time.time()-t3:.1f}s")

    # Build output
    output = {
        "features": verified,
        "demo_prompt": "Write a Python function that merges two sorted lists.",
        "backup_prompts": [
            "Implement a function that checks if a string is a palindrome.",
            "Write a function that counts word frequencies in a string.",
        ],
        "settings": {
            "temperature": 0.3,
            "max_new_tokens": 200,
            "slider_range": [-10, 10],
            "slider_step": 0.5,
        },
        "feature_labels_for_server": {},
    }

    # Build the FEATURE_LABELS dict for server.py
    for feat in verified:
        layer = str(feat["layer"])
        if layer not in output["feature_labels_for_server"]:
            output["feature_labels_for_server"][layer] = {}
        output["feature_labels_for_server"][layer][str(feat["feature_idx"])] = feat["label"]

    output_path = os.path.join(args.output_dir, "demo_features.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print(f"DONE — {len(verified)} verified features")
    print(f"Output: {output_path}")
    print(f"Total time: {time.time()-t0:.0f}s")
    print(f"{'='*70}")

    print("\nFEATURE_LABELS for server.py:")
    print("FEATURE_LABELS = {")
    for sae in saes:
        layer_feats = [f for f in verified if f["layer"] == sae.layer]
        print(f"    {sae.layer}: {{")
        for f in layer_feats:
            print(f"        {f['feature_idx']}: \"{f['label']}\",")
        print(f"    }},")
    print("}")


if __name__ == "__main__":
    main()
