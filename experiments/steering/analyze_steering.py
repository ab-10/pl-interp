"""Analyze steering experiment results: pass rate deltas across conditions."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

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


def _analyze_experiment(records: list[GenerationRecord]) -> dict:
    """Analyze a single experiment type (sae_steering or contrastive_steering)."""
    # Group by variant_id
    by_variant: dict[str, list[GenerationRecord]] = defaultdict(list)
    for r in records:
        by_variant[r.variant_id].append(r)

    # Baseline
    baseline_records = by_variant.get("baseline_no_steer", [])
    baseline_rate, baseline_passed, baseline_n = _compute_pass_rate(baseline_records)

    # Steering conditions
    conditions = []
    for variant_id, variant_records in sorted(by_variant.items()):
        condition, direction, alpha = _parse_variant(variant_id)
        if condition != "steer":
            continue

        rate, n_passed, n_total = _compute_pass_rate(variant_records)
        delta = rate - baseline_rate

        entry = {
            "direction": direction,
            "alpha": alpha,
            "pass_rate": round(rate, 4),
            "delta": round(delta, 4),
            "n_tasks": n_total,
            "n_passed": n_passed,
        }

        pvalue = _fisher_pvalue(baseline_passed, baseline_n, n_passed, n_total)
        if pvalue is not None:
            entry["fisher_p"] = round(pvalue, 6)

        conditions.append(entry)

    return {
        "baseline_pass_rate": round(baseline_rate, 4),
        "baseline_n": baseline_n,
        "baseline_passed": baseline_passed,
        "conditions": conditions,
    }


def _print_summary(experiment_type: str, result: dict) -> None:
    """Print a human-readable summary table."""
    label = experiment_type.replace("_", " ").title()
    baseline_n = result["baseline_n"]
    baseline_passed = result["baseline_passed"]
    baseline_rate = result["baseline_pass_rate"]

    print(f"\n=== {label} Results ===")
    print(f"Baseline: {baseline_passed}/{baseline_n} ({baseline_rate * 100:.1f}%)")
    print()

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

    print()


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
