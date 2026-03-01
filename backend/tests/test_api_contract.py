"""Unit tests for the Feature Steering API contract.

These tests use a mocked model/SAE so they run without a GPU.  They verify
request/response shapes, validation, and the steering code-path logic.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from .conftest import REGISTRY_FEATURES, SAE_D_SAE


# ── GET /features ──────────────────────────────────────────────────────────


class TestGetFeatures:
    def test_returns_200(self, client):
        resp = client.get("/features")
        assert resp.status_code == 200

    def test_returns_all_registry_entries(self, client):
        data = client.get("/features").json()
        returned_ids = {int(k) for k in data.keys()}
        assert returned_ids == REGISTRY_FEATURES

    def test_keys_are_strings(self, client):
        data = client.get("/features").json()
        assert all(isinstance(k, str) for k in data.keys())

    def test_values_are_strings(self, client):
        data = client.get("/features").json()
        assert all(isinstance(v, str) for v in data.values())


# ── POST /generate — request validation ────────────────────────────────────


class TestGenerateValidation:
    def test_missing_prompt_returns_422(self, client):
        resp = client.post("/generate", json={"features": []})
        assert resp.status_code == 422

    def test_empty_prompt_accepted(self, client):
        resp = client.post("/generate", json={"prompt": "", "features": []})
        assert resp.status_code == 200

    def test_features_defaults_to_empty(self, client):
        resp = client.post("/generate", json={"prompt": "hello"})
        assert resp.status_code == 200

    def test_temperature_defaults_to_0_3(self, client, mock_model):
        client.post("/generate", json={"prompt": "hello"})
        _, kwargs = mock_model.generate.call_args
        assert kwargs["temperature"] == 0.3


# ── POST /generate — baseline behaviour ────────────────────────────────────


class TestBaselineGeneration:
    def test_no_features_returns_identical_baseline_and_steered(self, client):
        data = client.post(
            "/generate", json={"prompt": "def fib(n):", "features": []}
        ).json()
        assert data["baseline"] == data["steered"]

    def test_zero_strength_features_treated_as_inactive(self, client):
        data = client.post(
            "/generate",
            json={
                "prompt": "def fib(n):",
                "features": [{"id": 6133, "strength": 0}],
            },
        ).json()
        assert data["baseline"] == data["steered"]

    def test_response_contains_baseline_and_steered_keys(self, client):
        data = client.post(
            "/generate", json={"prompt": "hello", "features": []}
        ).json()
        assert "baseline" in data
        assert "steered" in data


# ── POST /generate — steered behaviour ─────────────────────────────────────


class TestSteeredGeneration:
    def test_nonzero_strength_triggers_hooked_generate(self, client, mock_model):
        client.post(
            "/generate",
            json={
                "prompt": "hello",
                "features": [{"id": 6133, "strength": 100}],
            },
        )
        # model.generate called once for baseline, once inside hooks context
        assert mock_model.generate.call_count == 2

    def test_multiple_features_all_applied(self, client, mock_model):
        client.post(
            "/generate",
            json={
                "prompt": "hello",
                "features": [
                    {"id": 6133, "strength": 100},
                    {"id": 8019, "strength": -50},
                ],
            },
        )
        # Both features active → steered path taken
        assert mock_model.generate.call_count == 2
        mock_model.hooks.assert_called_once()

    def test_negative_strength_accepted(self, client):
        resp = client.post(
            "/generate",
            json={
                "prompt": "hello",
                "features": [{"id": 6133, "strength": -200}],
            },
        )
        assert resp.status_code == 200

    def test_large_strength_accepted(self, client):
        resp = client.post(
            "/generate",
            json={
                "prompt": "hello",
                "features": [{"id": 6133, "strength": 500}],
            },
        )
        assert resp.status_code == 200


# ── POST /generate — feature ID validation ─────────────────────────────────


class TestFeatureIdValidation:
    def test_negative_feature_id_rejected(self, client):
        resp = client.post(
            "/generate",
            json={
                "prompt": "hello",
                "features": [{"id": -1, "strength": 100}],
            },
        )
        assert resp.status_code == 500

    def test_feature_id_at_boundary_rejected(self, client):
        """Feature ID == d_sae (131072) is out of range."""
        resp = client.post(
            "/generate",
            json={
                "prompt": "hello",
                "features": [{"id": SAE_D_SAE, "strength": 100}],
            },
        )
        assert resp.status_code == 500

    def test_max_valid_feature_id_accepted(self, client):
        resp = client.post(
            "/generate",
            json={
                "prompt": "hello",
                "features": [{"id": SAE_D_SAE - 1, "strength": 100}],
            },
        )
        assert resp.status_code == 200

    def test_feature_id_zero_accepted(self, client):
        resp = client.post(
            "/generate",
            json={
                "prompt": "hello",
                "features": [{"id": 0, "strength": 100}],
            },
        )
        assert resp.status_code == 200


# ── POST /generate — temperature ───────────────────────────────────────────


class TestTemperature:
    def test_zero_temperature_disables_sampling(self, client, mock_model):
        client.post(
            "/generate",
            json={"prompt": "hello", "features": [], "temperature": 0},
        )
        _, kwargs = mock_model.generate.call_args
        assert kwargs["do_sample"] is False

    def test_positive_temperature_enables_sampling(self, client, mock_model):
        client.post(
            "/generate",
            json={"prompt": "hello", "features": [], "temperature": 0.7},
        )
        _, kwargs = mock_model.generate.call_args
        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == 0.7

    def test_max_new_tokens_is_200(self, client, mock_model):
        client.post("/generate", json={"prompt": "hello"})
        _, kwargs = mock_model.generate.call_args
        assert kwargs["max_new_tokens"] == 200
