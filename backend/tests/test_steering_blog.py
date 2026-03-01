"""Integration tests for Ministral 8B dual-layer SAE steering.

These tests require a running backend with the actual model + SAEs loaded.
Run with:
    BACKEND_URL=http://localhost:8000 pytest backend/tests/test_steering_blog.py -v
"""

import requests

from .conftest import BACKEND_URL, SAE_D_SAE, SAE_LAYERS


def _generate(prompt: str, features: list[dict], temperature: float = 0.0) -> dict:
    """Helper: call POST /generate and return JSON."""
    resp = requests.post(
        f"{BACKEND_URL}/generate",
        json={"prompt": prompt, "features": features, "temperature": temperature},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Endpoint smoke tests
# ---------------------------------------------------------------------------


class TestFeaturesEndpoint:
    def test_features_returns_per_layer_dict(self):
        resp = requests.get(f"{BACKEND_URL}/features", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        for layer in SAE_LAYERS:
            assert str(layer) in data

    def test_feature_ids_within_sae_range(self):
        data = requests.get(f"{BACKEND_URL}/features", timeout=10).json()
        for layer_str, labels in data.items():
            for key in labels:
                fid = int(key)
                assert 0 <= fid < SAE_D_SAE, f"Feature {fid} outside SAE range for layer {layer_str}"


class TestInfoEndpoint:
    def test_info_returns_model_and_saes(self):
        resp = requests.get(f"{BACKEND_URL}/info", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "Ministral" in data["model"]
        assert isinstance(data["saes"], dict)


# ---------------------------------------------------------------------------
# Baseline (unsteered) generation
# ---------------------------------------------------------------------------


class TestBaseline:
    def test_baseline_responds_coherently(self):
        """Without steering, the model should give a normal response."""
        result = _generate("Who are you?", [])
        baseline = result["baseline"]
        assert len(baseline) > 10, "Baseline output too short"
        assert result["baseline"] == result["steered"]

    def test_zero_strength_same_as_no_features(self):
        result = _generate(
            "Hello, how are you?",
            [{"id": 0, "layer": 18, "strength": 0}],
        )
        assert result["baseline"] == result["steered"]


# ---------------------------------------------------------------------------
# Steering on individual layers
# ---------------------------------------------------------------------------


class TestLayerSteering:
    def test_layer_18_steering_changes_output(self):
        """Steering on layer 18 should produce different output."""
        result = _generate(
            "def fibonacci(n):",
            [{"id": 0, "layer": 18, "strength": 5.0}],
        )
        assert result["steered"] != result["baseline"]

    def test_layer_27_steering_changes_output(self):
        """Steering on layer 27 should produce different output."""
        result = _generate(
            "def fibonacci(n):",
            [{"id": 0, "layer": 27, "strength": 5.0}],
        )
        assert result["steered"] != result["baseline"]

    def test_dual_layer_steering(self):
        """Steering on both layers simultaneously should work."""
        result = _generate(
            "def fibonacci(n):",
            [
                {"id": 0, "layer": 18, "strength": 5.0},
                {"id": 0, "layer": 27, "strength": 5.0},
            ],
        )
        assert result["steered"] != result["baseline"]


# ---------------------------------------------------------------------------
# Steering strength
# ---------------------------------------------------------------------------


class TestSteeringStrength:
    def test_negative_steering_differs_from_positive(self):
        """Opposite steering directions should produce different outputs."""
        pos = _generate(
            "def fibonacci(n):",
            [{"id": 0, "layer": 18, "strength": 5.0}],
        )
        neg = _generate(
            "def fibonacci(n):",
            [{"id": 0, "layer": 18, "strength": -5.0}],
        )
        assert pos["steered"] != neg["steered"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_feature_at_sae_boundary(self):
        """Feature ID d_sae-1 (max valid) should be accepted."""
        result = _generate(
            "hello",
            [{"id": SAE_D_SAE - 1, "layer": 18, "strength": 1.0}],
        )
        assert "steered" in result

    def test_feature_beyond_boundary_rejected(self):
        """Feature ID == d_sae is out of range."""
        resp = requests.post(
            f"{BACKEND_URL}/generate",
            json={
                "prompt": "hello",
                "features": [{"id": SAE_D_SAE, "layer": 18, "strength": 1.0}],
            },
            timeout=120,
        )
        assert resp.status_code == 500

    def test_deterministic_at_zero_temperature(self):
        """With temperature=0, same prompt+features should give same output."""
        a = _generate("def fib(n):", [{"id": 0, "layer": 18, "strength": 5.0}])
        b = _generate("def fib(n):", [{"id": 0, "layer": 18, "strength": 5.0}])
        assert a["steered"] == b["steered"]
        assert a["baseline"] == b["baseline"]
