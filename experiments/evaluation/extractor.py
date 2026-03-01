"""Code extraction from model outputs. Handles markdown blocks, bare code, and compliance checking."""

from __future__ import annotations

import re


def extract_code(text: str, expected_function_name: str | None = None) -> tuple[str, bool]:
    """Extract code from model output text.

    Strategy (in order):
    1. Markdown code block (```python...``` or ```...```)
    2. Bare function def
    3. Full text as fallback

    If multiple code blocks exist, prefer the one containing expected_function_name.

    Returns:
        (extracted_code, extraction_clean) — extraction_clean is True if a code block
        or bare function was found (not a full-text fallback on non-code text).
    """
    if not text or not text.strip():
        return ("", False)

    # Strategy 1: markdown code blocks
    # Match ```python ... ``` or ``` ... ```
    pattern = r"```(?:python)?\s*\n(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL)

    if blocks:
        if len(blocks) == 1:
            return (blocks[0].strip(), True)
        # Multiple blocks: prefer the one containing the expected function name
        if expected_function_name:
            for block in blocks:
                if f"def {expected_function_name}" in block:
                    return (block.strip(), True)
        # Default to first block
        return (blocks[0].strip(), True)

    # Strategy 2: bare function def (no markdown)
    # Find the first function definition and take everything from there
    func_pattern = r"^(def \w+.*)"
    match = re.search(func_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        code = match.group(1).strip()
        return (code, True)

    # Strategy 3: full text fallback
    return (text.strip(), False)


def check_compliance(extracted_code: str, expected_function_name: str) -> bool:
    """Check whether the extracted code defines the expected function.

    Returns True if the code contains a def statement with the expected function name.
    """
    if not extracted_code or not expected_function_name:
        return False
    pattern = rf"def {re.escape(expected_function_name)}\s*\("
    return bool(re.search(pattern, extracted_code))
