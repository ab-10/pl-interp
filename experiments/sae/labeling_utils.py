"""Shared utilities for SAE feature labeling with Claude on Bedrock.

Lightweight module with no heavy dependencies (no config, storage, tokenizer).
Used by both the batch labeling CLI and the backend server's relabel endpoint.
"""

from __future__ import annotations

import json
import logging
import re
import time

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

BEDROCK_MODEL_ID = "us.anthropic.claude-opus-4-6-v1"
BEDROCK_REGION = "us-east-1"


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
