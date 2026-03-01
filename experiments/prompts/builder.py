"""Prompt construction for HumanEval and MBPP tasks.

Builds the user message content for each task+variant combination.
The [INST] wrapping happens at generation time via tokenizer.apply_chat_template().
"""

from experiments.config import VARIANTS


def build_humaneval_prompt(task: dict, variant_id: str) -> str:
    """Build user message content for a HumanEval task.

    Format:
        {docstring_with_signature}

        Requirements:
        - Return only the function implementation.
        - No prints, no markdown, no explanation.
        {variant_instructions}

    Args:
        task: Normalized HumanEval task dict with 'prompt' and 'entry_point' keys.
        variant_id: One of the variant IDs from config.VARIANTS.

    Returns:
        The user message content string.
    """
    variant_instructions = VARIANTS[variant_id]

    lines = [
        task["prompt"].rstrip(),
        "",
        "Requirements:",
        "- Return only the function implementation.",
        "- No prints, no markdown, no explanation.",
    ]

    if variant_instructions:
        lines.append(f"- {variant_instructions}")

    return "\n".join(lines)


def build_mbpp_prompt(task: dict, variant_id: str) -> str:
    """Build user message content for an MBPP task.

    Format:
        {task_description}

        Write a Python function `{function_name}` that solves this.

        Requirements:
        - Return only the function implementation.
        - No prints, no markdown, no explanation.
        {variant_instructions}

    Args:
        task: Normalized MBPP task dict with 'prompt' and 'function_name' keys.
        variant_id: One of the variant IDs from config.VARIANTS.

    Returns:
        The user message content string.
    """
    variant_instructions = VARIANTS[variant_id]

    lines = [
        task["prompt"].rstrip(),
        "",
        f"Write a Python function `{task['function_name']}` that solves this.",
        "",
        "Requirements:",
        "- Return only the function implementation.",
        "- No prints, no markdown, no explanation.",
    ]

    if variant_instructions:
        lines.append(f"- {variant_instructions}")

    return "\n".join(lines)
