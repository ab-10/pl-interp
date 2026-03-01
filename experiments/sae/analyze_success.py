"""Analyze whether labeled SAE features contribute to code generation success.

Combines statistical signal (Cohen's d, fire rates) with LLM reasoning
to produce causal narratives about feature-success relationships.

Phase 1 — Stats: Scans activations through the SAE to compute per-feature
  fire rates and collect top-activating examples split by pass/fail.
Phase 2 — LLM: Sends each feature's stats + code contexts to Claude on
  Bedrock, asking whether the feature contributes to success.

Reads:  feature_labels.json, feature_stats.json, generation JSONLs, activations, SAE checkpoint
Writes: feature_success_analysis.json

Usage:
  python -m experiments.sae.analyze_success --model ministral-8b --layer 18
  python -m experiments.sae.analyze_success --model ministral-8b --layer 18 --stats-only
  python -m experiments.sae.analyze_success --model ministral-8b --layer 18 --dry-run
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments import config
from experiments.sae.analyze import TopExample, _load_sae
from experiments.sae.label_features import build_record_index, extract_code_context
from experiments.sae.labeling_utils import BEDROCK_REGION, call_bedrock
from experiments.storage.activation_store import ActivationReader
from experiments.storage.schema import read_records

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXAMPLES_PER_CLASS = 5
RATE_LIMIT_DELAY = 0.5
LLM_MAX_TOKENS = 500


# ---------------------------------------------------------------------------
# Prompt building & parsing
# ---------------------------------------------------------------------------

def build_success_prompt(
    feature_idx: int,
    label: str,
    description: str,
    layer: int,
    num_layers: int,
    cohens_d: float,
    mean_pass: float,
    mean_fail: float,
    fire_rate_pass: float,
    fire_rate_fail: float,
    pass_examples: list[dict],
    fail_examples: list[dict],
) -> str:
    """Build prompt asking Claude whether a feature contributes to success."""

    def _fmt(examples: list[dict]) -> str:
        if not examples:
            return "(no examples — feature rarely fires in this class)"
        parts = []
        for i, ex in enumerate(examples):
            parts.append(
                f"Example {i + 1} (activation: {ex['activation']:.4f}, task: {ex['task_id']}):\n"
                f"```\n{ex['code_context']}\n```"
            )
        return "\n\n".join(parts)

    position = "mid-network" if layer <= num_layers // 2 else "late-network"

    return (
        "You are analyzing whether an internal feature from a sparse autoencoder "
        "contributes to successful code generation.\n\n"
        "## Feature\n"
        f"- ID: #{feature_idx}\n"
        f"- Label: {label}\n"
        f"- Description: {description}\n"
        f"- Location: Layer {layer} of {num_layers} ({position})\n\n"
        "## Statistical Signal\n"
        f"- Cohen's d: {cohens_d:+.4f} (positive = more active in passing code)\n"
        f"- Mean activation (pass): {mean_pass:.4f}\n"
        f"- Mean activation (fail): {mean_fail:.4f}\n"
        f"- Fire rate (pass): {fire_rate_pass:.2%} of tokens\n"
        f"- Fire rate (fail): {fire_rate_fail:.2%} of tokens\n\n"
        "## Examples from PASSING code (the >>>token<<< is where the feature fires):\n\n"
        f"{_fmt(pass_examples)}\n\n"
        "## Examples from FAILING code (the >>>token<<< is where the feature fires):\n\n"
        f"{_fmt(fail_examples)}\n\n"
        "Based on the statistical signal and the code contexts, analyze whether this "
        "feature contributes to successful code generation.\n\n"
        "1. VERDICT: Does this feature contribute to success, is it neutral, or does "
        "it hinder? (contributes / neutral / hinders)\n"
        "2. MECHANISM: 1-2 sentences explaining the causal mechanism — WHY would this "
        "pattern be associated with correctness or incorrectness?\n"
        "3. CONFIDENCE: high / medium / low\n\n"
        "Format your response exactly as:\n"
        "VERDICT: <contributes|neutral|hinders>\n"
        "MECHANISM: <explanation>\n"
        "CONFIDENCE: <high|medium|low>"
    )


def parse_success_response(text: str) -> dict:
    """Parse VERDICT, MECHANISM, CONFIDENCE from LLM response."""
    verdict = "neutral"
    mechanism = ""
    confidence = "medium"

    m = re.search(r"VERDICT:\s*(contributes|neutral|hinders)", text, re.IGNORECASE)
    if m:
        verdict = m.group(1).strip().lower()

    m = re.search(r"MECHANISM:\s*(.+?)(?:\nCONFIDENCE|\Z)", text, re.DOTALL)
    if m:
        mechanism = m.group(1).strip()

    m = re.search(r"CONFIDENCE:\s*(high|medium|low)", text, re.IGNORECASE)
    if m:
        confidence = m.group(1).strip().lower()

    return {"verdict": verdict, "mechanism": mechanism, "confidence": confidence}


# ---------------------------------------------------------------------------
# Phase 1: Activation scanning (pass/fail split)
# ---------------------------------------------------------------------------

def scan_pass_fail_examples(
    labeled_features: set[int],
    sae_checkpoint: Path,
    generations_dir: Path,
    layer: int,
    examples_per_class: int = EXAMPLES_PER_CLASS,
    batch_size: int = 512,
    device: str = "cuda",
) -> dict:
    """Scan all records through the SAE. Collect pass/fail fire rates and top examples.

    Returns:
        {
            "total_tokens_pass": int,
            "total_tokens_fail": int,
            "features": {
                feat_idx: {
                    "fire_count_pass": int,
                    "fire_count_fail": int,
                    "pass_examples": [TopExample, ...],
                    "fail_examples": [TopExample, ...],
                }
            }
        }
    """
    logger.info("Loading SAE from %s", sae_checkpoint)
    sae = _load_sae(sae_checkpoint, device)

    logger.info("Loading generation records from %s", generations_dir)
    records = []
    for path in sorted(generations_dir.glob("*.jsonl")):
        records.extend(read_records(path))
    logger.info("Loaded %d records", len(records))

    # Per-feature counters and heaps
    fire_pass = {f: 0 for f in labeled_features}
    fire_fail = {f: 0 for f in labeled_features}
    pass_heaps: dict[int, list] = {f: [] for f in labeled_features}
    fail_heaps: dict[int, list] = {f: [] for f in labeled_features}
    total_pass = 0
    total_fail = 0

    readers: dict[str, ActivationReader] = {}
    layer_key = str(layer)

    for rec_idx, record in enumerate(records):
        if rec_idx % 2000 == 0:
            logger.info("Scanning record %d / %d", rec_idx, len(records))

        layer_info = record.activation_layers.get(layer_key)
        if layer_info is None:
            continue

        act_file = layer_info["file"]
        if act_file not in readers:
            readers[act_file] = ActivationReader(Path(act_file))

        act_np = readers[act_file].read(layer_info["offset"], layer_info["length"])
        num_tokens = act_np.shape[0]
        if num_tokens == 0:
            continue

        is_pass = record.passed
        if is_pass:
            total_pass += num_tokens
        else:
            total_fail += num_tokens

        for start in range(0, num_tokens, batch_size):
            end = min(start + batch_size, num_tokens)
            chunk = torch.from_numpy(act_np[start:end].astype(np.float32)).to(device)

            with torch.no_grad():
                _, _, info = sae(chunk)

            indices = info["topk_indices"].cpu().numpy()
            values = info["topk_values"].cpu().numpy()

            for tok in range(indices.shape[0]):
                for feat_idx, feat_val in zip(indices[tok], values[tok]):
                    feat_idx = int(feat_idx)
                    if feat_idx not in labeled_features:
                        continue

                    feat_val = float(feat_val)

                    if is_pass:
                        fire_pass[feat_idx] += 1
                    else:
                        fire_fail[feat_idx] += 1

                    example = TopExample(
                        value=feat_val,
                        task_id=record.task_id,
                        variant_id=record.variant_id,
                        run_id=record.run_id,
                        position=start + tok,
                    )
                    heap = pass_heaps[feat_idx] if is_pass else fail_heaps[feat_idx]
                    if len(heap) < examples_per_class:
                        heapq.heappush(heap, example)
                    elif feat_val > heap[0].value:
                        heapq.heapreplace(heap, example)

    logger.info("Scan complete: %d pass tokens, %d fail tokens", total_pass, total_fail)

    features_out = {}
    for f in labeled_features:
        features_out[f] = {
            "fire_count_pass": fire_pass[f],
            "fire_count_fail": fire_fail[f],
            "pass_examples": sorted(pass_heaps[f], key=lambda e: e.value, reverse=True),
            "fail_examples": sorted(fail_heaps[f], key=lambda e: e.value, reverse=True),
        }

    return {
        "total_tokens_pass": total_pass,
        "total_tokens_fail": total_fail,
        "features": features_out,
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def _write_output(path: Path, model_name: str, layer: int, results: dict) -> None:
    verdicts = [r.get("verdict", "") for r in results.values()]
    output = {
        "model": model_name,
        "layer": layer,
        "analyzer": "claude-opus-4-6",
        "features": results,
        "summary": {
            "total_analyzed": len(results),
            "contributes": sum(1 for v in verdicts if v == "contributes"),
            "neutral": sum(1 for v in verdicts if v == "neutral"),
            "hinders": sum(1 for v in verdicts if v == "hinders"),
            "pending": sum(1 for v in verdicts if v in ("error", "[dry-run]", "[stats-only]", "")),
        },
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def analyze_feature_success(
    feature_labels_path: Path,
    feature_stats_path: Path,
    sae_checkpoint: Path,
    generations_dir: Path,
    output_path: Path,
    model_name: str,
    layer: int,
    num_layers: int,
    examples_per_class: int = EXAMPLES_PER_CLASS,
    batch_size: int = 512,
    device: str = "cuda",
    stats_only: bool = False,
    dry_run: bool = False,
) -> Path:
    """Full feature-success analysis: scan activations, then ask Claude."""

    # --- Load labels ---
    with open(feature_labels_path) as f:
        labels_data = json.load(f)
    labeled_features = labels_data.get("features", {})
    feature_indices = {int(k) for k in labeled_features}
    logger.info("Loaded %d labeled features from %s", len(feature_indices), feature_labels_path)

    # --- Load existing stats (Cohen's d, means) ---
    stats_lookup: dict[int, dict] = {}
    if feature_stats_path and feature_stats_path.exists():
        with open(feature_stats_path) as f:
            stats_data = json.load(f)
        for feat in stats_data.get("features", []):
            stats_lookup[feat["feature_idx"]] = feat
        logger.info("Loaded stats for %d features", len(stats_lookup))

    # --- Phase 1: scan activations ---
    logger.info("=== Phase 1: Scanning activations ===")
    scan = scan_pass_fail_examples(
        labeled_features=feature_indices,
        sae_checkpoint=sae_checkpoint,
        generations_dir=generations_dir,
        layer=layer,
        examples_per_class=examples_per_class,
        batch_size=batch_size,
        device=device,
    )
    total_pass = scan["total_tokens_pass"]
    total_fail = scan["total_tokens_fail"]

    # --- Load tokenizer + record index for context extraction ---
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer for %s", config.MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)

    logger.info("Building record index...")
    record_index = build_record_index(generations_dir)

    # --- Bedrock client ---
    import boto3

    client = None
    if not stats_only and not dry_run:
        client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

    # --- Resumption ---
    existing: dict[str, dict] = {}
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f).get("features", {})
        logger.info("Loaded %d existing results for resumption", len(existing))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = dict(existing)

    # --- Phase 2: per-feature analysis ---
    logger.info("=== Phase 2: Analyzing %d features ===", len(feature_indices))
    sorted_features = sorted(feature_indices)

    for i, feat_idx in enumerate(sorted_features):
        feat_key = str(feat_idx)

        # Resumption: skip already-analyzed features (unless stats-only refresh)
        if not stats_only and feat_key in results:
            v = results[feat_key].get("verdict", "")
            if v and v not in ("error", "[dry-run]", "[stats-only]"):
                logger.info("Skipping feature %d (already analyzed)", feat_idx)
                continue

        label_info = labeled_features.get(feat_key, {})
        label = label_info.get("label", "unknown")
        description = label_info.get("description", "")

        stats = stats_lookup.get(feat_idx, {})
        cohens_d = stats.get("cohens_d", 0.0)
        mean_pass = stats.get("mean_pass", 0.0)
        mean_fail = stats.get("mean_fail", 0.0)

        scan_feat = scan["features"].get(feat_idx, {})
        fc_pass = scan_feat.get("fire_count_pass", 0)
        fc_fail = scan_feat.get("fire_count_fail", 0)
        fr_pass = fc_pass / total_pass if total_pass > 0 else 0.0
        fr_fail = fc_fail / total_fail if total_fail > 0 else 0.0

        # Extract code contexts
        pass_contexts = []
        for ex in scan_feat.get("pass_examples", []):
            ctx = extract_code_context(ex.to_dict(), record_index, tokenizer)
            if ctx:
                pass_contexts.append({"task_id": ex.task_id, "code_context": ctx, "activation": ex.value})

        fail_contexts = []
        for ex in scan_feat.get("fail_examples", []):
            ctx = extract_code_context(ex.to_dict(), record_index, tokenizer)
            if ctx:
                fail_contexts.append({"task_id": ex.task_id, "code_context": ctx, "activation": ex.value})

        logger.info(
            "Feature %d (%d/%d) [%s]: d=%+.3f, fire=%.2f%%/%.2f%%, ex=%d/%d",
            feat_idx, i + 1, len(sorted_features), label,
            cohens_d, fr_pass * 100, fr_fail * 100,
            len(pass_contexts), len(fail_contexts),
        )

        feature_result: dict = {
            "label": label,
            "description": description,
            "cohens_d": cohens_d,
            "mean_pass": mean_pass,
            "mean_fail": mean_fail,
            "fire_rate_pass": round(fr_pass, 6),
            "fire_rate_fail": round(fr_fail, 6),
            "fire_count_pass": fc_pass,
            "fire_count_fail": fc_fail,
            "pass_examples": pass_contexts,
            "fail_examples": fail_contexts,
        }

        # --- Stats-only mode ---
        if stats_only:
            feature_result["verdict"] = "[stats-only]"
            feature_result["mechanism"] = ""
            feature_result["llm_confidence"] = ""
            results[feat_key] = feature_result
            continue

        # --- Build LLM prompt ---
        prompt = build_success_prompt(
            feature_idx=feat_idx,
            label=label,
            description=description,
            layer=layer,
            num_layers=num_layers,
            cohens_d=cohens_d,
            mean_pass=mean_pass,
            mean_fail=mean_fail,
            fire_rate_pass=fr_pass,
            fire_rate_fail=fr_fail,
            pass_examples=pass_contexts,
            fail_examples=fail_contexts,
        )

        if dry_run:
            print(f"\n{'=' * 60}")
            print(f"FEATURE {feat_idx} ({i + 1}/{len(sorted_features)}) — {label}")
            print(f"{'=' * 60}")
            print(prompt)
            feature_result["verdict"] = "[dry-run]"
            feature_result["mechanism"] = ""
            feature_result["llm_confidence"] = ""
            results[feat_key] = feature_result
            continue

        # --- Call Bedrock ---
        try:
            response = call_bedrock(client, prompt, max_tokens=LLM_MAX_TOKENS)
            parsed = parse_success_response(response)
            feature_result["verdict"] = parsed["verdict"]
            feature_result["mechanism"] = parsed["mechanism"]
            feature_result["llm_confidence"] = parsed["confidence"]
            logger.info(
                "  -> %s (%s): %s",
                parsed["verdict"], parsed["confidence"], parsed["mechanism"][:100],
            )
        except Exception as e:
            logger.error("Failed to analyze feature %d: %s", feat_idx, e)
            feature_result["verdict"] = "error"
            feature_result["mechanism"] = str(e)
            feature_result["llm_confidence"] = "low"

        results[feat_key] = feature_result
        time.sleep(RATE_LIMIT_DELAY)

        # Crash-safe intermediate write
        _write_output(output_path, model_name, layer, results)

    # Final write
    _write_output(output_path, model_name, layer, results)
    logger.info("Wrote %d feature analyses to %s", len(results), output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Analyze whether SAE features contribute to code generation success.",
    )
    config.add_model_arg(parser)
    parser.add_argument("--layer", type=int, default=None, help="Layer number (default: primary capture layer).")
    parser.add_argument("--sae-checkpoint", type=Path, default=None, help="Path to trained SAE checkpoint (.pt).")
    parser.add_argument("--generations-dir", type=Path, default=None, help="Directory with generation JSONL shards.")
    parser.add_argument("--feature-labels", type=Path, default=None, help="Path to feature_labels.json.")
    parser.add_argument("--feature-stats", type=Path, default=None, help="Path to feature_stats.json.")
    parser.add_argument("--output", type=Path, default=None, help="Output path for feature_success_analysis.json.")
    parser.add_argument("--examples-per-class", type=int, default=EXAMPLES_PER_CLASS, help="Top examples per pass/fail class (default: 5).")
    parser.add_argument("--batch-size", type=int, default=512, help="SAE forward pass batch size (default: 512).")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device (default: cuda).")
    parser.add_argument("--stats-only", action="store_true", help="Compute stats without calling Bedrock.")
    parser.add_argument("--dry-run", action="store_true", help="Print LLM prompts without calling Bedrock.")
    args = parser.parse_args()
    config.apply_args(args)

    layer = args.layer or config.CAPTURE_LAYER
    generations_dir = args.generations_dir or config.GENERATIONS_DIR
    analysis_dir = config.ANALYSIS_DIR / f"layer_{layer}"
    sae_dir = config.SAE_DIR / f"layer_{layer}"

    feature_labels = args.feature_labels or analysis_dir / "feature_labels.json"
    feature_stats = args.feature_stats or analysis_dir / "feature_stats.json"
    output = args.output or analysis_dir / "feature_success_analysis.json"

    # Find SAE checkpoint
    sae_checkpoint = args.sae_checkpoint
    if sae_checkpoint is None:
        candidates = sorted(sae_dir.glob("*.pt"))
        if not candidates:
            logger.error("No SAE checkpoint found in %s. Use --sae-checkpoint.", sae_dir)
            return 1
        sae_checkpoint = candidates[-1]
        logger.info("Using SAE checkpoint: %s", sae_checkpoint)

    if not feature_labels.exists():
        logger.error("Feature labels not found: %s. Run label_features first.", feature_labels)
        return 1

    analyze_feature_success(
        feature_labels_path=feature_labels,
        feature_stats_path=feature_stats,
        sae_checkpoint=sae_checkpoint,
        generations_dir=generations_dir,
        output_path=output,
        model_name=config.MODEL_NAME,
        layer=layer,
        num_layers=config.MODEL_NUM_LAYERS,
        examples_per_class=args.examples_per_class,
        batch_size=args.batch_size,
        device=args.device,
        stats_only=args.stats_only,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
