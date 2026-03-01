"""Analyze steering experiment results: pass rate deltas, property density, and monotonicity."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments import config

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


def _wandb_enabled() -> bool:
    if _wandb is None:
        return False
    return os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1")
from experiments.storage.schema import GenerationRecord, read_records


# ---------------------------------------------------------------------------
# Property density patterns (adapted from SAE comparative eval spec)
# ---------------------------------------------------------------------------

DENSITY_PATTERNS: dict[str, list[str]] = {
    "type_annotations": [
        r":\s*(?:int|float|str|bool|list|dict|tuple|set|None|Any|Optional|Union|Callable)\b",
        r"->\s*(?:int|float|str|bool|list|dict|tuple|set|None|Any|Optional|Union)\b",
        r":\s*(?:List|Dict|Tuple|Set)\[",
    ],
    "error_handling": [
        r"\btry\b",
        r"\bexcept\b",
        r"\bcatch\b",
        r"\bfinally\b",
        r"\braise\b",
        r"\bthrow\b",
    ],
    "control_flow": [
        r"\bif\b",
        r"\belif\b",
        r"\belse\b",
        r"\bfor\b",
        r"\bwhile\b",
        r"\bbreak\b",
        r"\bcontinue\b",
        r"\bswitch\b",
        r"\bmatch\b",
    ],
    "decomposition": [
        r"\bdef\s+\w+",
        r"\bclass\s+\w+",
        r"\bfunction\s+\w+",
        r"\bimport\b",
        r"\bfrom\s+\w+\s+import\b",
    ],
    "functional_style": [
        r"\bmap\s*\(",
        r"\bfilter\s*\(",
        r"\breduce\s*\(",
        r"\blambda\b",
        r"\[.+\bfor\b.+\bin\b.+\]",
    ],
    "recursion": [
        r"\breturn\s+\w+\s*\(",
        r"\brecurs",
    ],
    "verbose_documentation": [
        r'"""',
        r"'''",
        r"#\s+\S",
        r"//\s+\S",
        r"/\*",
        r"\bArgs:\b",
        r"\bReturns:\b",
    ],
}


def compute_density(text: str, property_name: str) -> float:
    """Count regex pattern matches per line for a property. Returns 0.0 if unknown."""
    patterns = DENSITY_PATTERNS.get(property_name, [])
    if not patterns:
        return 0.0
    lines = text.strip().split("\n")
    total_lines = max(len(lines), 1)
    total_matches = sum(
        len(re.findall(p, text, re.MULTILINE | re.IGNORECASE))
        for p in patterns
    )
    return total_matches / total_lines


def compute_all_densities(text: str) -> dict[str, float]:
    """Compute density for every known property."""
    return {prop: compute_density(text, prop) for prop in DENSITY_PATTERNS}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_steering_records(steering_dir: Path) -> dict[str, list[GenerationRecord]]:
    """Load all steering JSONL files, grouped by experiment type.

    Experiment type is parsed from the filename prefix:
      sae_steering_*.jsonl       -> "sae_steering"
      contrastive_steering_*.jsonl -> "contrastive_steering"
    """
    groups: dict[str, list[GenerationRecord]] = defaultdict(list)

    for path in sorted(steering_dir.glob("*.jsonl")):
        name = path.stem  # e.g. "sae_steering_shard0"
        if name.startswith("sae_steering"):
            experiment_type = "sae_steering"
        elif name.startswith("contrastive_steering"):
            experiment_type = "contrastive_steering"
        else:
            logger.warning("Skipping unrecognized file: %s", path)
            continue

        records = read_records(path)
        groups[experiment_type].extend(records)
        logger.info("Loaded %d records from %s", len(records), path.name)

    return dict(groups)


# ---------------------------------------------------------------------------
# Variant parsing
# ---------------------------------------------------------------------------

_STEER_PATTERN = re.compile(r"^steer_(.+)_alpha_([-\d.]+)$")


def _parse_variant(variant_id: str) -> tuple[str, str | None, float | None]:
    """Parse variant_id into (condition, direction_name, alpha).

    Returns:
        ("baseline", None, None)  for "baseline_no_steer"
        ("steer", direction_name, alpha)  for "steer_{name}_alpha_{value}"
    """
    if variant_id == "baseline_no_steer":
        return ("baseline", None, None)

    m = _STEER_PATTERN.match(variant_id)
    if m:
        return ("steer", m.group(1), float(m.group(2)))

    return ("unknown", None, None)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _compute_pass_rate(records: list[GenerationRecord]) -> tuple[float, int, int]:
    """Return (pass_rate, n_passed, n_total)."""
    n = len(records)
    if n == 0:
        return (0.0, 0, 0)
    passed = sum(1 for r in records if r.passed)
    return (passed / n, passed, n)


def _fisher_pvalue(n_pass_a: int, n_total_a: int, n_pass_b: int, n_total_b: int) -> float | None:
    """Two-sided Fisher's exact test p-value, or None if scipy unavailable."""
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        return None

    table = [
        [n_pass_a, n_total_a - n_pass_a],
        [n_pass_b, n_total_b - n_pass_b],
    ]
    _, pvalue = fisher_exact(table)
    return float(pvalue)


def _compute_density_stats(records: list[GenerationRecord]) -> dict[str, float]:
    """Mean property density across all generated code in a set of records."""
    if not records:
        return {prop: 0.0 for prop in DENSITY_PATTERNS}
    densities_per_prop: dict[str, list[float]] = defaultdict(list)
    for r in records:
        text = r.extracted_code or r.generated_text
        if not text.strip():
            continue
        for prop, val in compute_all_densities(text).items():
            densities_per_prop[prop].append(val)
    return {
        prop: float(np.mean(vals)) if vals else 0.0
        for prop, vals in densities_per_prop.items()
    }


def _analyze_experiment(records: list[GenerationRecord]) -> dict:
    """Analyze a single experiment type (sae_steering or contrastive_steering)."""
    # Group by variant_id
    by_variant: dict[str, list[GenerationRecord]] = defaultdict(list)
    for r in records:
        by_variant[r.variant_id].append(r)

    # Baseline
    baseline_records = by_variant.get("baseline_no_steer", [])
    baseline_rate, baseline_passed, baseline_n = _compute_pass_rate(baseline_records)
    baseline_density = _compute_density_stats(baseline_records)

    # Steering conditions
    conditions = []
    for variant_id, variant_records in sorted(by_variant.items()):
        condition, direction, alpha = _parse_variant(variant_id)
        if condition != "steer":
            continue

        rate, n_passed, n_total = _compute_pass_rate(variant_records)
        delta = rate - baseline_rate
        density = _compute_density_stats(variant_records)

        entry = {
            "direction": direction,
            "alpha": alpha,
            "pass_rate": round(rate, 4),
            "delta": round(delta, 4),
            "n_tasks": n_total,
            "n_passed": n_passed,
            "density": {prop: round(v, 4) for prop, v in density.items()},
            "density_delta": {
                prop: round(v - baseline_density.get(prop, 0.0), 4)
                for prop, v in density.items()
            },
        }

        pvalue = _fisher_pvalue(baseline_passed, baseline_n, n_passed, n_total)
        if pvalue is not None:
            entry["fisher_p"] = round(pvalue, 6)

        conditions.append(entry)

    # --- Monotonicity analysis per direction ---
    # Group conditions by direction name
    by_direction: dict[str, list[dict]] = defaultdict(list)
    for c in conditions:
        by_direction[c["direction"]].append(c)

    monotonicity = {}
    for dir_name, dir_conds in by_direction.items():
        # For each density property, check monotonicity across alphas
        dir_mono = {}
        for prop in DENSITY_PATTERNS:
            neg_vals = [c["density"].get(prop, 0.0) for c in dir_conds if c["alpha"] < 0]
            pos_vals = [c["density"].get(prop, 0.0) for c in dir_conds if c["alpha"] > 0]
            bl_val = baseline_density.get(prop, 0.0)

            neg_avg = float(np.mean(neg_vals)) if neg_vals else bl_val
            pos_avg = float(np.mean(pos_vals)) if pos_vals else bl_val

            is_monotonic = (pos_avg > bl_val) and (bl_val > neg_avg)
            effect_size = pos_avg - neg_avg

            dir_mono[prop] = {
                "neg_avg": round(neg_avg, 4),
                "baseline": round(bl_val, 4),
                "pos_avg": round(pos_avg, 4),
                "is_monotonic": is_monotonic,
                "effect_size": round(effect_size, 4),
            }
        monotonicity[dir_name] = dir_mono

    return {
        "baseline_pass_rate": round(baseline_rate, 4),
        "baseline_n": baseline_n,
        "baseline_passed": baseline_passed,
        "baseline_density": {prop: round(v, 4) for prop, v in baseline_density.items()},
        "conditions": conditions,
        "monotonicity": monotonicity,
    }


def _print_summary(experiment_type: str, result: dict) -> None:
    """Print a human-readable summary table with pass rates, density, and monotonicity."""
    label = experiment_type.replace("_", " ").title()
    baseline_n = result["baseline_n"]
    baseline_passed = result["baseline_passed"]
    baseline_rate = result["baseline_pass_rate"]
    baseline_density = result.get("baseline_density", {})

    print(f"\n{'=' * 80}")
    print(f"  {label} Results")
    print(f"{'=' * 80}")
    print(f"Baseline: {baseline_passed}/{baseline_n} ({baseline_rate * 100:.1f}%)")
    if baseline_density:
        top_props = sorted(baseline_density.items(), key=lambda x: -x[1])[:4]
        dens_str = ", ".join(f"{p}={v:.3f}" for p, v in top_props)
        print(f"Baseline density: {dens_str}")
    print()

    # --- Pass rate table ---
    header = f"{'Direction':<25} {'Alpha':>6}   {'Pass Rate':>10}   {'Delta':>8}   {'N':>5}"
    print(header)
    print("-" * len(header))

    for c in result["conditions"]:
        direction = c["direction"] or "?"
        alpha_str = f"{c['alpha']:+.1f}" if c["alpha"] is not None else "?"
        rate_str = f"{c['pass_rate'] * 100:.1f}%"
        delta_str = f"{c['delta'] * 100:+.1f}%"
        n_str = str(c["n_tasks"])

        sig = ""
        if "fisher_p" in c and c["fisher_p"] < 0.05:
            sig = " *"

        print(f"{direction:<25} {alpha_str:>6}   {rate_str:>10}   {delta_str:>8}   {n_str:>5}{sig}")

    # --- Monotonicity results ---
    monotonicity = result.get("monotonicity", {})
    if monotonicity:
        print(f"\n--- Monotonicity Analysis (pos_avg > baseline > neg_avg) ---")
        for dir_name, props in sorted(monotonicity.items()):
            mono_props = [p for p, v in props.items() if v["is_monotonic"]]
            top_effects = sorted(props.items(), key=lambda x: -abs(x[1]["effect_size"]))[:3]

            if mono_props:
                print(f"  {dir_name}: MONOTONIC for {', '.join(mono_props)}")
            else:
                print(f"  {dir_name}: no monotonic properties")

            for prop, v in top_effects:
                marker = " <-- MONOTONIC" if v["is_monotonic"] else ""
                print(f"    {prop:<25} neg={v['neg_avg']:.3f}  bl={v['baseline']:.3f}  "
                      f"pos={v['pos_avg']:.3f}  effect={v['effect_size']:+.4f}{marker}")

    # --- Contamination check ---
    _print_contamination(result)

    print()


def _print_contamination(result: dict) -> None:
    """Check which directions move the same density properties (cross-direction contamination)."""
    monotonicity = result.get("monotonicity", {})
    if not monotonicity:
        return

    # For each direction, find the property with the largest |effect_size|
    dir_top_property: dict[str, str] = {}
    for dir_name, props in monotonicity.items():
        if "random" in dir_name:
            continue
        best = max(props.items(), key=lambda x: abs(x[1]["effect_size"]))
        dir_top_property[dir_name] = best[0]

    # Check if multiple real directions primarily affect the same property
    prop_counts = Counter(dir_top_property.values())
    contaminated = {prop for prop, count in prop_counts.items() if count > 1}

    if contaminated:
        print(f"\n--- Contamination Warning ---")
        for prop in contaminated:
            dirs = [d for d, p in dir_top_property.items() if p == prop]
            print(f"  '{prop}' is the top-affected property for: {', '.join(dirs)}")
        print("  These directions may not be capturing distinct properties.")
    else:
        if dir_top_property:
            print(f"\n--- Contamination Check: CLEAN ---")
            for d, p in sorted(dir_top_property.items()):
                print(f"  {d} -> primarily affects '{p}'")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_steering(steering_dir: Path, output_dir: Path) -> Path:
    """Compute pass rate deltas for all steering experiments and export results.

    Args:
        steering_dir: Directory containing steering JSONL result files.
        output_dir: Directory for output JSON.

    Returns:
        Path to the written steering_results.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading steering results from %s", steering_dir)
    groups = _load_steering_records(steering_dir)

    if not groups:
        logger.warning("No steering result files found in %s", steering_dir)

    results = {}
    for experiment_type, records in sorted(groups.items()):
        logger.info("Analyzing %s (%d records)", experiment_type, len(records))
        result = _analyze_experiment(records)
        results[experiment_type] = result
        _print_summary(experiment_type, result)

    out_path = output_dir / "steering_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Wrote %s", out_path)

    # --- wandb summary ---
    if _wandb_enabled():
        _wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name="steering-analysis",
        )
        for exp_type, result in results.items():
            _wandb.summary[f"{exp_type}/baseline_pass_rate"] = result["baseline_pass_rate"]
            for cond in result["conditions"]:
                key = f"{exp_type}/{cond['direction']}_alpha_{cond['alpha']}"
                _wandb.summary[f"{key}/pass_rate"] = cond["pass_rate"]
                _wandb.summary[f"{key}/delta"] = cond["delta"]
                if "fisher_p" in cond:
                    _wandb.summary[f"{key}/fisher_p"] = cond["fisher_p"]
        _wandb.finish()

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze steering experiment results (pass rate deltas).",
    )
    parser.add_argument(
        "--steering-dir",
        type=Path,
        default=config.STEERING_DIR,
        help=f"Directory with steering JSONL files (default: {config.STEERING_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.ANALYSIS_DIR,
        help=f"Directory for output files (default: {config.ANALYSIS_DIR}).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    analyze_steering(
        steering_dir=args.steering_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
