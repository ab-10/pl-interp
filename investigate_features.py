#!/usr/bin/env python3
"""
Investigate SAE features 13176 and 16290 from layer 18 of Ministral-8B.
Run on h100-dev-box-5:
    python3 investigate_features.py
"""

import json
import sys
from pathlib import Path

FEATURES = [13176, 16290]
LAYER_18_DIR = Path("/scratch/ministral-8b/analysis/layer_18")
LAYER_27_DIR = Path("/scratch/ministral-8b/analysis/layer_27")

# ── Helper ───────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}\n")


# ── 1. Probe weights ────────────────────────────────────────────────────────

section("1. PROBE WEIGHTS FOR FEATURES 13176 & 16290")

probe_dir_path = LAYER_18_DIR / "probe_direction.pt"
probe_stats_path = LAYER_18_DIR / "probe_stats.json"

if probe_dir_path.exists():
    try:
        import torch
        probe_direction = torch.load(probe_dir_path, map_location="cpu", weights_only=True)
        print(f"Loaded probe_direction.pt  shape={probe_direction.shape}  dtype={probe_direction.dtype}")
        for fidx in FEATURES:
            if fidx < probe_direction.shape[-1]:
                w = probe_direction[fidx] if probe_direction.dim() == 1 else probe_direction[..., fidx]
                print(f"  Feature {fidx}: probe weight = {w.item():.6f}")
            else:
                print(f"  Feature {fidx}: index out of range (max={probe_direction.shape[-1]-1})")
    except Exception as e:
        print(f"Error loading probe_direction.pt: {e}")
else:
    print(f"probe_direction.pt NOT found at {probe_dir_path}")
    # Also check alternative locations
    for alt in [
        LAYER_18_DIR / "probe_weights.pt",
        LAYER_18_DIR / "probe.pt",
        LAYER_18_DIR / "linear_probe.pt",
    ]:
        if alt.exists():
            print(f"  (found alternative: {alt})")

if probe_stats_path.exists():
    probe_stats = load_json(probe_stats_path)
    print(f"\nprobe_stats.json keys: {list(probe_stats.keys())}")
    if "top_features" in probe_stats:
        top = probe_stats["top_features"]
        print(f"top_features has {len(top)} entries")
        for entry in top:
            idx = entry.get("feature_idx") or entry.get("index") or entry.get("feature")
            if idx in FEATURES:
                print(f"  Feature {idx} found in top_features: {json.dumps(entry, indent=4)}")
    # Check if there's a full weight list
    for key in ["weights", "all_weights", "weight_vector", "coefficients"]:
        if key in probe_stats:
            vec = probe_stats[key]
            print(f"\nFound '{key}' (length={len(vec)})")
            for fidx in FEATURES:
                if fidx < len(vec):
                    print(f"  Feature {fidx}: weight = {vec[fidx]}")
else:
    print(f"probe_stats.json NOT found at {probe_stats_path}")


# ── 2. Variant breakdown ────────────────────────────────────────────────────

section("2. VARIANT BREAKDOWN FOR BOTH FEATURES")

feat_stats_path = LAYER_18_DIR / "feature_stats.json"
if not feat_stats_path.exists():
    print(f"feature_stats.json NOT found at {feat_stats_path}")
    sys.exit(1)

feat_stats = load_json(feat_stats_path)
print(f"feature_stats.json type: {type(feat_stats).__name__}")

# Determine structure: could be dict keyed by str(idx), or a list
def get_feature(data, idx):
    """Try multiple access patterns."""
    if isinstance(data, dict):
        for key in [str(idx), idx, f"feature_{idx}"]:
            if key in data:
                return data[key]
        # Maybe nested under 'features'
        if "features" in data:
            return get_feature(data["features"], idx)
    elif isinstance(data, list) and idx < len(data):
        return data[idx]
    return None

for fidx in FEATURES:
    fdata = get_feature(feat_stats, fidx)
    if fdata is None:
        print(f"Feature {fidx}: NOT FOUND in feature_stats.json")
        continue

    print(f"\n--- Feature {fidx} ---")
    # Print variant_means / variant breakdown
    for vkey in ["variant_means", "variant_breakdown", "variants", "per_variant"]:
        if vkey in fdata:
            print(f"  {vkey}:")
            vdata = fdata[vkey]
            if isinstance(vdata, dict):
                for vname, vval in sorted(vdata.items(), key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0):
                    print(f"    {vname}: {vval}")
            elif isinstance(vdata, list):
                for item in vdata:
                    print(f"    {item}")


# ── 3. Top activating examples ──────────────────────────────────────────────

section("3. TOP 5 ACTIVATING EXAMPLES FOR BOTH FEATURES")

for fidx in FEATURES:
    fdata = get_feature(feat_stats, fidx)
    if fdata is None:
        print(f"Feature {fidx}: no data")
        continue

    print(f"\n--- Feature {fidx} ---")
    examples = None
    for ekey in ["top_examples", "top_activations", "examples", "activating_examples", "max_activations"]:
        if ekey in fdata:
            examples = fdata[ekey]
            break

    if examples is None:
        print("  No top examples field found.")
        print(f"  Available keys: {list(fdata.keys()) if isinstance(fdata, dict) else 'N/A'}")
        continue

    top5 = examples[:5] if isinstance(examples, list) else examples
    for i, ex in enumerate(top5 if isinstance(top5, list) else [top5]):
        task_id = ex.get("task_id", ex.get("id", "?"))
        variant_id = ex.get("variant_id", ex.get("variant", "?"))
        position = ex.get("position", ex.get("pos", ex.get("token_pos", "?")))
        activation = ex.get("activation", ex.get("act", ex.get("value", "?")))
        print(f"  [{i+1}] task_id={task_id}  variant_id={variant_id}  position={position}  activation={activation}")


# ── 4. Pass/fail correlation ────────────────────────────────────────────────

section("4. PASS/FAIL CORRELATION")

for fidx in FEATURES:
    fdata = get_feature(feat_stats, fidx)
    if fdata is None:
        print(f"Feature {fidx}: no data")
        continue

    print(f"\n--- Feature {fidx} ---")
    mean_pass = fdata.get("mean_pass", fdata.get("pass_mean"))
    mean_fail = fdata.get("mean_fail", fdata.get("fail_mean"))

    if mean_pass is not None and mean_fail is not None:
        diff = mean_pass - mean_fail
        # Estimate pass-rate proxy: if we treat activation as a signal
        print(f"  mean_pass  = {mean_pass:.6f}")
        print(f"  mean_fail  = {mean_fail:.6f}")
        print(f"  difference = {diff:+.6f}  (pass - fail)")
        if mean_fail != 0:
            pct = (diff / abs(mean_fail)) * 100
            print(f"  relative   = {pct:+.1f}% vs fail baseline")
    else:
        print(f"  mean_pass/mean_fail not found. Available keys: {list(fdata.keys()) if isinstance(fdata, dict) else 'N/A'}")

    # Also check for explicit pass_rate fields
    for prkey in ["pass_rate_high", "pass_rate_low", "pass_rate_when_active", "pass_rate_when_inactive",
                   "pass_rate_q4", "pass_rate_q1", "pass_rate_difference"]:
        if prkey in fdata:
            print(f"  {prkey} = {fdata[prkey]}")


# ── 5. Cohen's d ranking ────────────────────────────────────────────────────

section("5. COHEN'S D RANKING AMONG ALL 16384 FEATURES")

# Collect |cohen's_d| for all features
cohens = []

if isinstance(feat_stats, dict):
    items = feat_stats.items() if "features" not in feat_stats else feat_stats["features"].items() if isinstance(feat_stats.get("features"), dict) else enumerate(feat_stats.get("features", []))
else:
    items = enumerate(feat_stats)

for key, fdata in items:
    if not isinstance(fdata, dict):
        continue
    cd = fdata.get("cohens_d", fdata.get("cohen_d", fdata.get("effect_size")))
    if cd is not None:
        try:
            fidx = int(key)
        except (ValueError, TypeError):
            fidx = key
        cohens.append((fidx, abs(cd), cd))

if cohens:
    # Sort by |cohen's_d| descending
    cohens.sort(key=lambda x: x[1], reverse=True)
    total = len(cohens)
    print(f"Total features with Cohen's d: {total}")

    for rank, (fidx, abs_cd, raw_cd) in enumerate(cohens, 1):
        if fidx in FEATURES:
            pct = (rank / total) * 100
            print(f"  Feature {fidx}: |d|={abs_cd:.4f}  (raw d={raw_cd:.4f})  rank={rank}/{total}  (top {pct:.2f}%)")

    # Also show top 10 for context
    print(f"\n  Top 10 by |Cohen's d| for context:")
    for rank, (fidx, abs_cd, raw_cd) in enumerate(cohens[:10], 1):
        marker = " <---" if fidx in FEATURES else ""
        print(f"    #{rank}: feature {fidx}  |d|={abs_cd:.4f}  d={raw_cd:.4f}{marker}")
else:
    print("No Cohen's d values found in feature_stats.json.")
    # Show a sample entry to understand structure
    sample_key = next(iter(feat_stats if isinstance(feat_stats, dict) else range(1)), None)
    if sample_key is not None:
        sample = feat_stats[sample_key] if isinstance(feat_stats, dict) else feat_stats[0]
        if isinstance(sample, dict):
            print(f"  Sample entry keys: {list(sample.keys())}")


# ── 6. Cross-layer check (layer 27) ─────────────────────────────────────────

section("6. CROSS-LAYER CHECK: SAME INDICES IN LAYER 27")

feat_stats_27_path = LAYER_27_DIR / "feature_stats.json"
if not feat_stats_27_path.exists():
    print(f"feature_stats.json NOT found at {feat_stats_27_path}")
    # Check what layers exist
    analysis_root = Path("/scratch/ministral-8b/analysis")
    if analysis_root.exists():
        layers = sorted([d.name for d in analysis_root.iterdir() if d.is_dir()])
        print(f"  Available layer dirs: {layers}")
else:
    feat_stats_27 = load_json(feat_stats_27_path)
    print(f"Loaded layer 27 feature_stats.json")

    for fidx in FEATURES:
        fdata_27 = get_feature(feat_stats_27, fidx)
        if fdata_27 is None:
            print(f"\n  Feature {fidx}: NOT FOUND in layer 27")
            continue

        print(f"\n--- Feature {fidx} (Layer 27) ---")
        # Print key stats for comparison
        for k in ["cohens_d", "cohen_d", "effect_size", "mean_pass", "mean_fail",
                   "mean_activation", "sparsity", "variant_means", "variant_breakdown"]:
            if k in fdata_27:
                val = fdata_27[k]
                if isinstance(val, dict):
                    print(f"  {k}:")
                    for vk, vv in val.items():
                        print(f"    {vk}: {vv}")
                else:
                    print(f"  {k} = {val}")

        # Compare with layer 18
        fdata_18 = get_feature(feat_stats, fidx)
        if fdata_18:
            cd18 = fdata_18.get("cohens_d", fdata_18.get("cohen_d"))
            cd27 = fdata_27.get("cohens_d", fdata_27.get("cohen_d"))
            if cd18 is not None and cd27 is not None:
                print(f"  Layer 18 d={cd18:.4f}  vs  Layer 27 d={cd27:.4f}")


print(f"\n{'='*72}")
print("  INVESTIGATION COMPLETE")
print(f"{'='*72}\n")
