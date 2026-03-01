"""Automated SAE feature labeling using Claude on Bedrock.

For each top feature (by probe weight or Cohen's d), extracts code contexts
from the highest-activating examples and asks Claude to describe what the
feature detects.

Reads:  probe_stats.json, feature_stats.json, generation JSONL shards
Writes: feature_labels.json

Usage:
  python -m experiments.sae.label_features --model ministral-8b --layer 18 --top-n 50
  python -m experiments.sae.label_features --model ministral-8b --layer 18 --top-n 5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from transformers import AutoTokenizer

from experiments import config
from experiments.storage.schema import GenerationRecord, read_records

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BEDROCK_MODEL_ID = "anthropic.claude-opus-4-6-v1"
BEDROCK_REGION = "us-east-1"
CONTEXT_WINDOW = 40  # tokens before + after the activation position
RATE_LIMIT_DELAY = 0.5  # seconds between Bedrock calls
MAX_EXAMPLES_PER_FEATURE = 5


# ---------------------------------------------------------------------------
# Record index
# ---------------------------------------------------------------------------

def build_record_index(
    generations_dir: Path,
) -> dict[tuple[str, str, int], GenerationRecord]:
    """Build lookup index: (task_id, variant_id, run_id) -> GenerationRecord."""
    index: dict[tuple[str, str, int], GenerationRecord] = {}
    for path in sorted(generations_dir.glob("*.jsonl")):
        for record in read_records(path):
            key = (record.task_id, record.variant_id, record.run_id)
            index[key] = record
    logger.info("Built record index: %d records", len(index))
    return index


# ---------------------------------------------------------------------------
# Code context extraction
# ---------------------------------------------------------------------------

def extract_code_context(
    example: dict,
    record_index: dict[tuple[str, str, int], GenerationRecord],
    tokenizer,
    context_tokens: int = CONTEXT_WINDOW,
) -> str | None:
    """Extract code context around the activating token position.

    Returns a string with the activating token marked with >>>token<<<,
    or None if the record is not found.
    """
    key = (example["task_id"], example["variant_id"], example["run_id"])
    record = record_index.get(key)
    if record is None or not record.gen_token_ids:
        return None

    position = example["position"]
    token_ids = record.gen_token_ids

    if position >= len(token_ids):
        return None

    # Context window around the activation position
    start = max(0, position - context_tokens // 2)
    end = min(len(token_ids), position + context_tokens // 2)

    before_text = tokenizer.decode(token_ids[start:position], skip_special_tokens=True)
    active_text = tokenizer.decode([token_ids[position]], skip_special_tokens=False)
    after_text = tokenizer.decode(token_ids[position + 1 : end], skip_special_tokens=True)

    return f"{before_text}>>>{active_text}<<<{after_text}"


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features_to_label(
    probe_stats_path: Path | None,
    feature_stats_path: Path | None,
    top_n: int = 50,
) -> list[int]:
    """Select top features to label, preferring probe-weighted features.

    Returns deduplicated list of feature indices.
    """
    selected: list[int] = []
    seen: set[int] = set()

    # Primary: probe-weighted features
    if probe_stats_path and probe_stats_path.exists():
        with open(probe_stats_path) as f:
            probe_stats = json.load(f)
        for entry in probe_stats.get("top_features", []):
            idx = entry["feature_idx"]
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)

    # Secondary: top Cohen's d features from feature_stats
    if feature_stats_path and feature_stats_path.exists() and len(selected) < top_n:
        with open(feature_stats_path) as f:
            feature_stats = json.load(f)
        features = feature_stats.get("features", [])
        ranked = sorted(features, key=lambda f: abs(f.get("cohens_d", 0)), reverse=True)
        for feat in ranked:
            idx = feat["feature_idx"]
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
            if len(selected) >= top_n:
                break

    return selected[:top_n]


def get_feature_examples(
    feature_idx: int,
    feature_stats_path: Path | None,
) -> list[dict]:
    """Get top_examples for a specific feature from feature_stats.json."""
    if not feature_stats_path or not feature_stats_path.exists():
        return []
    with open(feature_stats_path) as f:
        data = json.load(f)
    for feat in data.get("features", []):
        if feat["feature_idx"] == feature_idx:
            return feat.get("top_examples", [])
    return []


# ---------------------------------------------------------------------------
# Bedrock LLM
# ---------------------------------------------------------------------------

def call_bedrock(
    client,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 300,
    max_retries: int = 3,
) -> str:
    """Call Claude Opus 4.6 on Bedrock with exponential backoff."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    })

    for attempt in range(max_retries):
        try:
            response = client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            return result["content"][0]["text"]
        except ClientError as e:
            if "ThrottlingException" in str(e) and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning("Throttled, retrying in %ds...", wait)
                time.sleep(wait)
            else:
                raise

    return ""


def build_labeling_prompt(feature_idx: int, examples: list[dict]) -> str:
    """Build the prompt for Claude to label a feature."""
    examples_text = "\n\n---\n\n".join(
        f"Example {i + 1} (activation strength: {ex['activation']:.4f}, task: {ex['task_id']}):\n"
        f"```\n{ex['code_context']}\n```\n"
        f"The token marked with >>> <<< is where the feature activated."
        for i, ex in enumerate(examples)
    )

    return (
        f"You are analyzing an internal feature (feature #{feature_idx}) from a sparse autoencoder "
        f"trained on a code-generating language model (Ministral-8B). Below are code snippets where "
        f"this feature activated most strongly. The token wrapped in >>>token<<< is the exact "
        f"position where the feature fired.\n\n"
        f"{examples_text}\n\n"
        f"Based on the patterns across these examples, provide:\n\n"
        f"1. LABEL: A concise label (3-5 words) for what this feature detects\n"
        f"2. DESCRIPTION: 1-2 sentences explaining what code pattern, syntactic element, or "
        f"semantic property this feature responds to\n"
        f"3. CONFIDENCE: high, medium, or low -- based on how consistent the pattern is across examples\n\n"
        f"Format your response exactly as:\n"
        f"LABEL: <label>\n"
        f"DESCRIPTION: <description>\n"
        f"CONFIDENCE: <high|medium|low>"
    )


def parse_label_response(response_text: str) -> dict:
    """Parse LABEL, DESCRIPTION, CONFIDENCE from LLM response."""
    label = ""
    description = ""
    confidence = "medium"

    label_match = re.search(r"LABEL:\s*(.+?)(?:\n|$)", response_text)
    desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?:\n|$)", response_text)
    conf_match = re.search(r"CONFIDENCE:\s*(high|medium|low)", response_text, re.IGNORECASE)

    if label_match:
        label = label_match.group(1).strip()
    if desc_match:
        description = desc_match.group(1).strip()
    if conf_match:
        confidence = conf_match.group(1).strip().lower()

    return {"label": label, "description": description, "confidence": confidence}


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def label_features(
    generations_dir: Path,
    feature_stats_path: Path | None,
    probe_stats_path: Path | None,
    output_path: Path,
    model_name: str,
    layer: int,
    top_n: int = 50,
    max_examples: int = MAX_EXAMPLES_PER_FEATURE,
    context_tokens: int = CONTEXT_WINDOW,
    dry_run: bool = False,
) -> Path:
    """Label top SAE features using Claude on Bedrock.

    Args:
        generations_dir: Directory with generation JSONL shards.
        feature_stats_path: Path to feature_stats.json (top_examples per feature).
        probe_stats_path: Path to probe_stats.json (top probe-weighted features).
        output_path: Where to write feature_labels.json.
        model_name: Model name for metadata.
        layer: Layer number for metadata.
        top_n: Number of features to label.
        max_examples: Max examples per feature to show the LLM.
        context_tokens: Token window around activation position.
        dry_run: Print prompts without calling Bedrock.

    Returns:
        Path to the written feature_labels.json.
    """
    # Load existing labels for resumption
    existing: dict[str, dict] = {}
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f).get("features", {})
        logger.info("Loaded %d existing labels for resumption", len(existing))

    # Select features
    feature_indices = select_features_to_label(probe_stats_path, feature_stats_path, top_n)
    logger.info("Selected %d features to label", len(feature_indices))

    # Load tokenizer (CPU only)
    logger.info("Loading tokenizer for %s", config.MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)

    # Build record index
    logger.info("Building record index from %s", generations_dir)
    record_index = build_record_index(generations_dir)

    # Initialize Bedrock client
    client = None
    if not dry_run:
        client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = dict(existing)

    for i, feat_idx in enumerate(feature_indices):
        feat_key = str(feat_idx)

        # Skip already-labeled features (resumption)
        if feat_key in results and results[feat_key].get("label") and results[feat_key]["label"] != "error":
            logger.info("Skipping feature %d (already labeled)", feat_idx)
            continue

        logger.info("Labeling feature %d (%d/%d)", feat_idx, i + 1, len(feature_indices))

        # Get top examples
        raw_examples = get_feature_examples(feat_idx, feature_stats_path)
        if not raw_examples:
            logger.warning("No examples found for feature %d", feat_idx)
            results[feat_key] = {"label": "no examples", "description": "", "confidence": "low", "examples": []}
            continue

        # Extract code contexts
        labeled_examples = []
        for ex in raw_examples[:max_examples]:
            context = extract_code_context(ex, record_index, tokenizer, context_tokens)
            if context:
                labeled_examples.append({
                    "task_id": ex["task_id"],
                    "code_context": context,
                    "activation": ex["value"],
                })

        if not labeled_examples:
            logger.warning("Could not extract contexts for feature %d", feat_idx)
            results[feat_key] = {"label": "no contexts", "description": "", "confidence": "low", "examples": []}
            continue

        # Build prompt
        prompt = build_labeling_prompt(feat_idx, labeled_examples)

        if dry_run:
            print(f"\n{'=' * 60}")
            print(f"FEATURE {feat_idx} ({i + 1}/{len(feature_indices)})")
            print(f"{'=' * 60}")
            print(prompt)
            print()
            results[feat_key] = {
                "label": "[dry-run]",
                "description": "[dry-run]",
                "confidence": "low",
                "examples": labeled_examples,
            }
            continue

        # Call Bedrock
        try:
            response_text = call_bedrock(client, prompt)
            parsed = parse_label_response(response_text)
            results[feat_key] = {
                **parsed,
                "examples": labeled_examples,
            }
            logger.info("  -> %s: %s", parsed["label"], parsed["description"][:80])
        except Exception as e:
            logger.error("Failed to label feature %d: %s", feat_idx, e)
            results[feat_key] = {
                "label": "error",
                "description": str(e),
                "confidence": "low",
                "examples": labeled_examples,
            }

        # Rate limit
        time.sleep(RATE_LIMIT_DELAY)

        # Write intermediate results after each feature (crash-safe)
        output = {
            "model": model_name,
            "layer": layer,
            "labeler": "claude-opus-4-6",
            "features": results,
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

    # Final write
    output = {
        "model": model_name,
        "layer": layer,
        "labeler": "claude-opus-4-6",
        "features": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Wrote %d feature labels to %s", len(results), output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Label SAE features using Claude Opus 4.6 on Bedrock.",
    )
    config.add_model_arg(parser)
    parser.add_argument(
        "--generations-dir", type=Path, default=None,
        help="Directory containing generation JSONL shards.",
    )
    parser.add_argument(
        "--feature-stats", type=Path, default=None,
        help="Path to feature_stats.json.",
    )
    parser.add_argument(
        "--probe-stats", type=Path, default=None,
        help="Path to probe_stats.json.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for feature_labels.json.",
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Layer number (default: primary capture layer).",
    )
    parser.add_argument(
        "--top-n", type=int, default=50,
        help="Number of features to label (default: 50).",
    )
    parser.add_argument(
        "--max-examples", type=int, default=MAX_EXAMPLES_PER_FEATURE,
        help=f"Max examples per feature (default: {MAX_EXAMPLES_PER_FEATURE}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompts without calling Bedrock.",
    )
    args = parser.parse_args()
    config.set_model(args.model)

    layer = args.layer or config.CAPTURE_LAYER
    generations_dir = args.generations_dir or config.GENERATIONS_DIR
    analysis_dir = config.ANALYSIS_DIR / f"layer_{layer}"
    feature_stats = args.feature_stats or analysis_dir / "feature_stats.json"
    probe_stats = args.probe_stats or analysis_dir / "probe_stats.json"
    output = args.output or analysis_dir / "feature_labels.json"

    label_features(
        generations_dir=generations_dir,
        feature_stats_path=feature_stats if feature_stats.exists() else None,
        probe_stats_path=probe_stats if probe_stats.exists() else None,
        output_path=output,
        model_name=config.MODEL_NAME,
        layer=layer,
        top_n=args.top_n,
        max_examples=args.max_examples,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
