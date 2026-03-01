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

from backend.server_custom_sae import _build_feature_label, _find_monotonicity, _get_monotonicity_data

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
    def test_monotonic_shows_direction(self, mistral_7b_candidates, mistral_7b_steering):
        """Feature 40693: monotonic control_flow -> label should say 'Increases Control Flow'."""
        candidate = next(
            c for c in mistral_7b_candidates["candidates"]
            if c["feature_idx"] == 40693
        )
        label = _build_feature_label(candidate, mistral_7b_steering)
        assert "Control Flow" in label
        assert label.startswith("Increases") or label.startswith("Decreases")

    def test_non_monotonic_shows_property_name(self, mistral_7b_candidates, mistral_7b_steering):
        """Feature 123379: non-monotonic -> label should be just the property display name."""
        candidate = next(
            c for c in mistral_7b_candidates["candidates"]
            if c["feature_idx"] == 123379
        )
        label = _build_feature_label(candidate, mistral_7b_steering)
        # Non-monotonic: should just show the property name without Increases/Decreases
        assert not label.startswith("Increases")
        assert not label.startswith("Decreases")

    def test_fallback_without_steering_data(self, mistral_7b_candidates):
        """Without steering results, falls back to title-cased variant name."""
        candidate = next(
            c for c in mistral_7b_candidates["candidates"]
            if c["feature_idx"] == 40693
        )
        label = _build_feature_label(candidate, {})
        assert label == "Error Handling"

    def test_labels_are_strings(self, mistral_7b_candidates, mistral_7b_steering):
        for c in mistral_7b_candidates["candidates"]:
            label = _build_feature_label(c, mistral_7b_steering)
            assert isinstance(label, str)
            assert len(label) > 3

    def test_labels_are_human_readable(self, mistral_7b_candidates, mistral_7b_steering):
        """Labels should not contain snake_case or technical jargon."""
        for c in mistral_7b_candidates["candidates"]:
            label = _build_feature_label(c, mistral_7b_steering)
            assert "_" not in label, f"Label contains underscore: {label}"
            assert "d=" not in label, f"Label contains technical notation: {label}"
            assert "effect=" not in label, f"Label contains technical notation: {label}"


# ---------------------------------------------------------------------------
# _get_monotonicity_data
# ---------------------------------------------------------------------------


class TestGetMonotonicityData:
    def test_returns_full_property_dict(self, mistral_7b_steering):
        result = _get_monotonicity_data("40693", mistral_7b_steering)
        assert result is not None
        assert isinstance(result, dict)
        # Should have multiple properties
        assert len(result) >= 1
        # Each property should have expected fields
        for prop_data in result.values():
            assert "effect_size" in prop_data
            assert "is_monotonic" in prop_data

    def test_returns_none_for_unknown(self, mistral_7b_steering):
        result = _get_monotonicity_data("999999", mistral_7b_steering)
        assert result is None

    def test_returns_none_for_empty_analysis(self):
        result = _get_monotonicity_data("40693", {})
        assert result is None
