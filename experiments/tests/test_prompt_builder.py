"""Tests for prompt construction, variant templates, and universal contract."""

import pytest

from experiments.config import VARIANTS, VARIANT_IDS
from experiments.prompts.builder import build_humaneval_prompt, build_mbpp_prompt


# --- Universal contract appears in all variants ---

UNIVERSAL_CONTRACT_FRAGMENTS = [
    "Return only the function implementation",
    "No prints",
    "no markdown",
    "no explanation",
]


class TestUniversalContract:
    """Universal output contract must appear in every variant prompt."""

    @pytest.mark.parametrize("variant_id", VARIANT_IDS)
    def test_humaneval_contract(self, sample_humaneval_task, variant_id):
        prompt = build_humaneval_prompt(sample_humaneval_task, variant_id)
        for fragment in UNIVERSAL_CONTRACT_FRAGMENTS:
            assert fragment in prompt, (
                f"Missing contract fragment '{fragment}' in HumanEval {variant_id} prompt"
            )

    @pytest.mark.parametrize("variant_id", VARIANT_IDS)
    def test_mbpp_contract(self, sample_mbpp_task, variant_id):
        prompt = build_mbpp_prompt(sample_mbpp_task, variant_id)
        for fragment in UNIVERSAL_CONTRACT_FRAGMENTS:
            assert fragment in prompt, (
                f"Missing contract fragment '{fragment}' in MBPP {variant_id} prompt"
            )


# --- Each variant contains its structural instruction ---

class TestVariantInstructions:
    """Each non-baseline variant must include its specific structural instruction."""

    @pytest.mark.parametrize("variant_id", [v for v in VARIANT_IDS if v != "baseline"])
    def test_humaneval_variant_instruction(self, sample_humaneval_task, variant_id):
        prompt = build_humaneval_prompt(sample_humaneval_task, variant_id)
        instruction = VARIANTS[variant_id]
        assert instruction in prompt, (
            f"Variant instruction not found in HumanEval {variant_id} prompt"
        )

    @pytest.mark.parametrize("variant_id", [v for v in VARIANT_IDS if v != "baseline"])
    def test_mbpp_variant_instruction(self, sample_mbpp_task, variant_id):
        prompt = build_mbpp_prompt(sample_mbpp_task, variant_id)
        instruction = VARIANTS[variant_id]
        assert instruction in prompt, (
            f"Variant instruction not found in MBPP {variant_id} prompt"
        )


# --- Baseline has no extra instructions beyond the contract ---

class TestBaseline:
    """Baseline variant should have only the universal contract, no extra instructions."""

    def test_humaneval_baseline_no_extra(self, sample_humaneval_task):
        prompt = build_humaneval_prompt(sample_humaneval_task, "baseline")
        # Baseline should not contain any non-baseline variant instructions
        for vid, instruction in VARIANTS.items():
            if vid != "baseline" and instruction:
                assert instruction not in prompt, (
                    f"Baseline HumanEval prompt should not contain {vid} instruction"
                )

    def test_mbpp_baseline_no_extra(self, sample_mbpp_task):
        prompt = build_mbpp_prompt(sample_mbpp_task, "baseline")
        for vid, instruction in VARIANTS.items():
            if vid != "baseline" and instruction:
                assert instruction not in prompt, (
                    f"Baseline MBPP prompt should not contain {vid} instruction"
                )


# --- HumanEval and MBPP produce different formats ---

class TestFormatDifferences:
    """HumanEval and MBPP tasks produce different prompt formats."""

    def test_different_formats(self, sample_humaneval_task, sample_mbpp_task):
        he_prompt = build_humaneval_prompt(sample_humaneval_task, "baseline")
        mbpp_prompt = build_mbpp_prompt(sample_mbpp_task, "baseline")
        # They should be different strings
        assert he_prompt != mbpp_prompt

    def test_humaneval_contains_signature(self, sample_humaneval_task):
        prompt = build_humaneval_prompt(sample_humaneval_task, "baseline")
        # HumanEval prompt should contain the function signature from the dataset
        assert sample_humaneval_task["entry_point"] in prompt

    def test_mbpp_contains_function_directive(self, sample_mbpp_task):
        prompt = build_mbpp_prompt(sample_mbpp_task, "baseline")
        # MBPP format includes "Write a Python function `{function_name}`"
        assert f"Write a Python function `{sample_mbpp_task['function_name']}`" in prompt

    def test_humaneval_does_not_have_write_directive(self, sample_humaneval_task):
        prompt = build_humaneval_prompt(sample_humaneval_task, "baseline")
        assert "Write a Python function" not in prompt

    def test_mbpp_does_not_have_docstring(self, sample_mbpp_task, sample_humaneval_task):
        mbpp_prompt = build_mbpp_prompt(sample_mbpp_task, "baseline")
        # MBPP should not contain HumanEval-style docstring markers
        assert '"""' not in mbpp_prompt


# --- Prompt includes the task description ---

class TestTaskDescription:
    """Prompts must include the actual task description/problem."""

    def test_humaneval_includes_task(self, sample_humaneval_task):
        prompt = build_humaneval_prompt(sample_humaneval_task, "baseline")
        # The HumanEval prompt field (docstring+signature) should be in the output
        assert "has_close_elements" in prompt
        assert "Check if in given list of numbers" in prompt

    def test_mbpp_includes_task(self, sample_mbpp_task):
        prompt = build_mbpp_prompt(sample_mbpp_task, "baseline")
        # The MBPP text (task description) should be in the output
        assert sample_mbpp_task["prompt"] in prompt


# --- Determinism: same inputs produce same output ---

class TestDeterminism:
    """Prompt construction is deterministic — same task + variant = same prompt."""

    def test_humaneval_deterministic(self, sample_humaneval_task):
        p1 = build_humaneval_prompt(sample_humaneval_task, "typed")
        p2 = build_humaneval_prompt(sample_humaneval_task, "typed")
        assert p1 == p2

    def test_mbpp_deterministic(self, sample_mbpp_task):
        p1 = build_mbpp_prompt(sample_mbpp_task, "invariants")
        p2 = build_mbpp_prompt(sample_mbpp_task, "invariants")
        assert p1 == p2
