"""Unit tests for custom SAE server feature label generation.

Tests _build_feature_label and _find_monotonicity with real artifact data.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Stub heavy ML libraries before importing server module
import sys
for mod_name in ("torch", "torch.cuda", "torch.nn", "transformers"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.server_custom_sae import _build_feature_label, _find_monotonicity

ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "artifacts"
MISTRAL_7B = ARTIFACTS / "mistral-7b"


@pytest.fixture()
def mistral_7b_candidates():
    with open(MISTRAL_7B / "feature_candidates.json") as f:
        return json.load(f)


@pytest.fixture()
def mistral_7b_steering():
    with open(MISTRAL_7B / "steering_results.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# _find_monotonicity
# ---------------------------------------------------------------------------


class TestFindMonotonicity:
    def test_finds_monotonic_direction(self, mistral_7b_steering):
        result = _find_monotonicity("40693", mistral_7b_steering)
        assert result is not None
        prop, effect, is_mono = result
        assert prop == "control_flow"
        assert is_mono is True
        assert effect > 0.1  # effect=+0.179

    def test_finds_non_monotonic_direction(self, mistral_7b_steering):
        result = _find_monotonicity("128287", mistral_7b_steering)
        assert result is not None
        prop, effect, is_mono = result
        assert prop == "control_flow"
        assert is_mono is False

    def test_returns_none_for_unknown(self, mistral_7b_steering):
        result = _find_monotonicity("999999", mistral_7b_steering)
        assert result is None

    def test_returns_none_for_empty_analysis(self):
        result = _find_monotonicity("40693", {})
        assert result is None


# ---------------------------------------------------------------------------
# _build_feature_label
# ---------------------------------------------------------------------------


class TestBuildFeatureLabel:
    def test_monotonic_with_redirection(self, mistral_7b_candidates, mistral_7b_steering):
        """Feature 40693: selected as error_handling, actually steers control_flow."""
        candidate = next(
            c for c in mistral_7b_candidates["candidates"]
            if c["feature_idx"] == 40693
        )
        label = _build_feature_label(candidate, mistral_7b_steering)
        assert "error_handling" in label
        assert "control_flow" in label
        assert "monotonic" in label
        assert "->" in label  # shows redirection

    def test_different_property_shows_arrow(self, mistral_7b_candidates, mistral_7b_steering):
        """Feature 123379: control_flow -> type_annotations (different, so has arrow)."""
        candidate = next(
            c for c in mistral_7b_candidates["candidates"]
            if c["feature_idx"] == 123379
        )
        label = _build_feature_label(candidate, mistral_7b_steering)
        assert "->" in label

    def test_fallback_without_steering_data(self, mistral_7b_candidates):
        """Without steering results, falls back to Cohen's d label."""
        candidate = next(
            c for c in mistral_7b_candidates["candidates"]
            if c["feature_idx"] == 40693
        )
        label = _build_feature_label(candidate, {})
        assert "error_handling" in label
        assert "d=" in label
        assert "monotonic" not in label

    def test_labels_are_strings(self, mistral_7b_candidates, mistral_7b_steering):
        for c in mistral_7b_candidates["candidates"]:
            label = _build_feature_label(c, mistral_7b_steering)
            assert isinstance(label, str)
            assert len(label) > 5

    def test_effect_size_in_label(self, mistral_7b_candidates, mistral_7b_steering):
        """Effect size should appear in the label when analysis is available."""
        candidate = next(
            c for c in mistral_7b_candidates["candidates"]
            if c["feature_idx"] == 40693
        )
        label = _build_feature_label(candidate, mistral_7b_steering)
        assert "effect=" in label
