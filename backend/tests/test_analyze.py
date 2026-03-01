"""Unit tests for the POST /analyze endpoint.

These tests verify request validation and schema. Full computation tests
require a GPU; see the Test Infrastructure Note in the plan.
"""

from .conftest import SAE_D_SAE


# ── Request validation ────────────────────────────────────────────────────


class TestAnalyzeValidation:
    def test_missing_prompt_returns_422(self, client):
        resp = client.post("/analyze", json={"feature_id": 100})
        assert resp.status_code == 422

    def test_missing_feature_id_returns_422(self, client):
        resp = client.post("/analyze", json={"prompt": "hello"})
        assert resp.status_code == 422

    def test_negative_feature_id_returns_400(self, client):
        resp = client.post(
            "/analyze",
            json={"prompt": "hello", "feature_id": -1, "steering": []},
        )
        assert resp.status_code == 400

    def test_feature_id_at_boundary_returns_400(self, client):
        resp = client.post(
            "/analyze",
            json={"prompt": "hello", "feature_id": SAE_D_SAE, "steering": []},
        )
        assert resp.status_code == 400

    def test_valid_feature_id_accepted(self, client):
        """A valid feature_id should pass validation (may fail later in
        computation when torch is mocked, but should not return 400/422)."""
        resp = client.post(
            "/analyze",
            json={"prompt": "hello", "feature_id": 100, "steering": []},
        )
        # 200 if computation succeeds, 500 if torch mocks cause issues
        # but NOT 400 or 422
        assert resp.status_code not in (400, 422)

    def test_steering_defaults_to_empty(self, client):
        resp = client.post(
            "/analyze", json={"prompt": "hello", "feature_id": 100}
        )
        assert resp.status_code not in (400, 422)

    def test_invalid_steering_feature_id_returns_400(self, client):
        resp = client.post(
            "/analyze",
            json={
                "prompt": "hello",
                "feature_id": 100,
                "steering": [{"id": SAE_D_SAE, "strength": 5.0}],
            },
        )
        assert resp.status_code == 400
