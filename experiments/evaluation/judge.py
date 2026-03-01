"""Failure classification and test script builders for HumanEval and MBPP."""

from __future__ import annotations

from experiments.storage.schema import compute_error_hash


def classify_failure(
    passed: bool, stderr: str, exit_code: int, extracted_code: str
) -> tuple[str, str, str]:
    """Classify execution outcome into a failure category.

    Priority order:
    1. extraction_fail (empty code)
    2. pass (exit code 0)
    3. timeout (from executor)
    4. syntax_error (SyntaxError in stderr)
    5. type_error (TypeError in stderr)
    6. wrong_answer (AssertionError in stderr)
    7. runtime_error (anything else)

    Returns:
        (failure_category, error_message, error_hash)
    """
    # 1. Extraction failure — empty code never ran
    if not extracted_code or not extracted_code.strip():
        return ("extraction_fail", "no code extracted", compute_error_hash("no code extracted"))

    # 2. Pass
    if passed and exit_code == 0:
        return ("pass", "", "")

    # 3. Timeout
    if stderr == "timeout" and exit_code == -1:
        return ("timeout", "timeout", compute_error_hash("timeout"))

    # 4-7. Parse stderr for specific error types
    error_message = stderr.strip() if stderr else f"exit_code={exit_code}"

    if "SyntaxError" in error_message:
        return ("syntax_error", error_message, compute_error_hash(error_message))

    if "TypeError" in error_message:
        return ("type_error", error_message, compute_error_hash(error_message))

    if "AssertionError" in error_message or "AssertionError" in error_message:
        return ("wrong_answer", error_message, compute_error_hash(error_message))

    return ("runtime_error", error_message, compute_error_hash(error_message))


def build_humaneval_test_script(
    extracted_code: str, test: str, entry_point: str
) -> str:
    """Build an executable test script for a HumanEval task.

    The HumanEval test string defines a check(candidate) function and calls it.
    We define the generated function, then run check(entry_point).

    Args:
        extracted_code: The generated function code.
        test: The HumanEval check() function definition and invocation.
        entry_point: The function name to pass to check().

    Returns:
        A complete Python script string.
    """
    return f"""{extracted_code}

{test}

check({entry_point})
"""


def build_mbpp_test_script(
    extracted_code: str, test_list: list[str], test_setup_code: str = ""
) -> str:
    """Build an executable test script for an MBPP task.

    Args:
        extracted_code: The generated function code.
        test_list: List of assertion strings (e.g., "assert func(args) == expected").
        test_setup_code: Optional setup code (imports, helpers).

    Returns:
        A complete Python script string.
    """
    parts = []
    if test_setup_code:
        parts.append(test_setup_code)
    parts.append(extracted_code)
    parts.append("\n".join(test_list))
    return "\n\n".join(parts) + "\n"
