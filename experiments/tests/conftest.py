"""Shared fixtures for experiment tests."""

import pytest

from experiments.config import VARIANT_IDS


@pytest.fixture
def sample_humaneval_task() -> dict:
    """A realistic sample HumanEval task dict."""
    return {
        "task_id": "humaneval_0",
        "dataset": "humaneval",
        "prompt": (
            'from typing import List\n\n\n'
            'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n'
            '    """ Check if in given list of numbers, are any two numbers closer to each other than\n'
            '    given threshold.\n'
            '    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n'
            '    False\n'
            '    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n'
            '    True\n'
            '    """\n'
        ),
        "entry_point": "has_close_elements",
        "test": (
            'def check(candidate):\n'
            '    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n'
            '    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n'
        ),
        "canonical_solution": (
            '    for idx, elem in enumerate(numbers):\n'
            '        for idx2, elem2 in enumerate(numbers):\n'
            '            if idx != idx2:\n'
            '                distance = abs(elem - elem2)\n'
            '                if distance < threshold:\n'
            '                    return True\n'
            '    return False\n'
        ),
    }


@pytest.fixture
def sample_mbpp_task() -> dict:
    """A realistic sample MBPP task dict."""
    return {
        "task_id": "mbpp_11",
        "dataset": "mbpp",
        "prompt": "Write a function to remove all characters except letters and numbers.",
        "function_name": "remove_dirty_chars",
        "test_list": [
            "assert remove_dirty_chars('ty##!@', 'jkl%^&') == 'tyjkl'",
            "assert remove_dirty_chars('is#$', 'txt%^&') == 'istxt'",
            "assert remove_dirty_chars('h!@', 'wor##') == 'hwor'",
        ],
        "test_setup_code": "",
    }


@pytest.fixture
def all_variant_ids() -> list[str]:
    """All variant IDs from config."""
    return VARIANT_IDS
