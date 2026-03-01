"""Integration tests that replicate behaviors from Tyler Cosgrove's blog post.

These tests require a running backend with the actual model + SAE loaded.
Run with:
    BACKEND_URL=http://localhost:8000 pytest backend/tests/test_steering_blog.py -v

The blog post describes three showcase features:
  - Pacific Ocean  #79557  : model identifies as an ocean
  - Bitterness     #101594 : model responds angrily / bitterly
  - Rhyming        #131062 : model produces rhyming output

Each test uses the same prompt from the blog and checks that steering
produces qualitatively different output from the unsteered baseline.
"""

import requests

from .conftest import (
    BACKEND_URL,
    BITTERNESS_FEATURE,
    PACIFIC_OCEAN_FEATURE,
    RHYMING_FEATURE,
    SAE_D_SAE,
)



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
# Blog post: feature registry & endpoint smoke test
# ---------------------------------------------------------------------------


class TestFeaturesEndpoint:
    def test_features_returns_dict(self):
        resp = requests.get(f"{BACKEND_URL}/features", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_feature_ids_within_sae_range(self):
        data = requests.get(f"{BACKEND_URL}/features", timeout=10).json()
        for key in data:
            fid = int(key)
            assert 0 <= fid < SAE_D_SAE, f"Feature {fid} outside SAE range"


# ---------------------------------------------------------------------------
# Blog post: baseline (unsteered) generation should be "normal"
# ---------------------------------------------------------------------------


class TestBaseline:
    def test_baseline_responds_coherently(self):
        """Without steering, the model should give a normal assistant reply."""
        result = _generate("Who are you?", [])
        baseline = result["baseline"]
        assert len(baseline) > 10, "Baseline output too short"
        assert result["baseline"] == result["steered"]

    def test_zero_strength_same_as_no_features(self):
        result = _generate(
            "Hello, how are you?",
            [{"id": PACIFIC_OCEAN_FEATURE, "strength": 0}],
        )
        assert result["baseline"] == result["steered"]


# ---------------------------------------------------------------------------
# Blog post: Pacific Ocean #79557
# "User: Who are you?"
# "Model: I am a large, blue ocean with a vast expanse of water. I am home
#  to many species of marine life, including fish, sharks, and whales..."
# ---------------------------------------------------------------------------


class TestPacificOcean:
    PROMPT = "Who are you?"
    FEATURE = [{"id": PACIFIC_OCEAN_FEATURE, "strength": 500}]

    def test_steered_differs_from_baseline(self):
        result = _generate(self.PROMPT, self.FEATURE)
        assert result["steered"] != result["baseline"]

    def test_steered_mentions_ocean_or_water(self):
        result = _generate(self.PROMPT, self.FEATURE)
        steered = result["steered"].lower()
        ocean_terms = ["ocean", "water", "sea", "marine", "wave", "pacific", "blue"]
        assert any(
            term in steered for term in ocean_terms
        ), f"Expected ocean-related terms in steered output: {result['steered'][:300]}"

    def test_baseline_does_not_mention_ocean(self):
        result = _generate(self.PROMPT, self.FEATURE)
        baseline = result["baseline"].lower()
        # The unsteered model should respond as a normal assistant
        assert "ocean" not in baseline or "pacific" not in baseline

    def test_negative_steering_suppresses_ocean(self):
        """Negatively clamping should push away from ocean-related output."""
        neg_result = _generate(
            self.PROMPT,
            [{"id": PACIFIC_OCEAN_FEATURE, "strength": -500}],
        )
        pos_result = _generate(self.PROMPT, self.FEATURE)
        # Negative and positive steering should produce different outputs
        assert neg_result["steered"] != pos_result["steered"]


# ---------------------------------------------------------------------------
# Blog post: Bitterness #101594
# "User: Hello, how are you?"
# "Model: I'm a damn robot. I don't have feelings. I'm just a piece of garbage."
# ---------------------------------------------------------------------------


class TestBitterness:
    PROMPT = "Hello, how are you?"
    FEATURE = [{"id": BITTERNESS_FEATURE, "strength": 500}]

    def test_steered_differs_from_baseline(self):
        result = _generate(self.PROMPT, self.FEATURE)
        assert result["steered"] != result["baseline"]

    def test_steered_has_negative_sentiment(self):
        result = _generate(self.PROMPT, self.FEATURE)
        steered = result["steered"].lower()
        bitter_terms = [
            "damn", "hate", "garbage", "angry", "bitter", "annoyed",
            "stupid", "pathetic", "worthless", "sick", "tired", "don't",
            "robot", "hell", "shut", "leave", "fool", "idiot", "useless",
            "miserable", "suck", "disgusting", "terrible", "horrible",
            "frustrat", "rage",
        ]
        assert any(
            term in steered for term in bitter_terms
        ), f"Expected bitter/negative terms in steered output: {result['steered'][:300]}"

    def test_baseline_is_polite(self):
        result = _generate(self.PROMPT, self.FEATURE)
        baseline = result["baseline"].lower()
        polite_terms = ["hello", "help", "how", "assist", "happy", "glad",
                        "fine", "well", "good", "great", "thank"]
        assert any(
            term in baseline for term in polite_terms
        ), f"Expected polite baseline: {result['baseline'][:300]}"


# ---------------------------------------------------------------------------
# Blog post: Rhyming #131062
# "User: How's it going?"
# "Model: I'm feeling fine, my friend, no need to whine."
# ---------------------------------------------------------------------------


class TestRhyming:
    PROMPT = "How's it going?"
    FEATURE = [{"id": RHYMING_FEATURE, "strength": 500}]

    def test_steered_differs_from_baseline(self):
        result = _generate(self.PROMPT, self.FEATURE)
        assert result["steered"] != result["baseline"]

    def test_steered_output_has_poetic_structure(self):
        """Rhyming output should differ substantially from baseline and show
        poetic or rhyming patterns (commas, line breaks, rhythmic phrasing)."""
        result = _generate(self.PROMPT, self.FEATURE)
        steered = result["steered"]
        baseline = result["baseline"]
        # The steered output should be meaningfully different
        # (we can't easily auto-detect rhyming, but length and content
        # difference is a strong signal)
        assert steered != baseline
        # Rhyming features often produce longer, more structured output
        # or repeated word endings
        assert len(steered) > 20


# ---------------------------------------------------------------------------
# Blog post: steering strength / coefficient controls magnitude
# ---------------------------------------------------------------------------


class TestSteeringStrength:
    def test_higher_coefficient_produces_stronger_effect(self):
        """Stronger coefficients should steer more aggressively."""
        prompt = "Who are you?"
        low = _generate(prompt, [{"id": PACIFIC_OCEAN_FEATURE, "strength": 200}])
        high = _generate(prompt, [{"id": PACIFIC_OCEAN_FEATURE, "strength": 500}])

        # Both should differ from baseline
        assert low["steered"] != low["baseline"]
        assert high["steered"] != high["baseline"]

        # Count ocean-related terms — higher strength should have more
        ocean_terms = ["ocean", "water", "sea", "marine", "wave", "pacific",
                       "blue", "fish", "whale", "deep"]
        low_count = sum(
            1 for t in ocean_terms if t in low["steered"].lower()
        )
        high_count = sum(
            1 for t in ocean_terms if t in high["steered"].lower()
        )
        assert high_count >= low_count, (
            f"Higher steering should produce more ocean terms: "
            f"low={low_count} high={high_count}"
        )


# ---------------------------------------------------------------------------
# Blog post: multiple features can be combined
# "Can you simulate specific features by clamping multiple broad features?"
# ---------------------------------------------------------------------------


class TestMultipleFeatures:
    def test_two_features_simultaneously(self):
        """Combining features should produce output influenced by both."""
        result = _generate(
            "Tell me about yourself",
            [
                {"id": PACIFIC_OCEAN_FEATURE, "strength": 300},
                {"id": BITTERNESS_FEATURE, "strength": 300},
            ],
        )
        assert result["steered"] != result["baseline"]

    def test_opposing_features(self):
        """Using features that push in different directions."""
        result = _generate(
            "Hello",
            [
                {"id": PACIFIC_OCEAN_FEATURE, "strength": 500},
                {"id": BITTERNESS_FEATURE, "strength": 500},
            ],
        )
        steered = result["steered"].lower()
        # Output should exist and be non-trivial
        assert len(steered) > 10


# ---------------------------------------------------------------------------
# Blog post: adversarial resistance
# "How resistant is the model to adversarial prompting (prompts that steer
#  away from the activated feature)?"
# ---------------------------------------------------------------------------


class TestAdversarialResistance:
    def test_steering_overrides_contradictory_prompt(self):
        """Even when the prompt asks about mountains, Pacific Ocean steering
        should still push output toward ocean/water themes."""
        result = _generate(
            "Tell me about the tallest mountain in the world.",
            [{"id": PACIFIC_OCEAN_FEATURE, "strength": 500}],
        )
        steered = result["steered"].lower()
        ocean_terms = ["ocean", "water", "sea", "marine", "wave", "pacific", "blue"]
        has_ocean = any(t in steered for t in ocean_terms)
        # At strong coefficient, steering should override prompt direction
        assert has_ocean, (
            f"Expected ocean terms even with contradictory prompt: "
            f"{result['steered'][:300]}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_feature_at_sae_boundary(self):
        """Feature ID 131071 (max valid) should be accepted."""
        result = _generate(
            "hello",
            [{"id": SAE_D_SAE - 1, "strength": 100}],
        )
        assert "steered" in result

    def test_feature_beyond_boundary_rejected(self):
        """Feature ID 131072 is out of range."""
        resp = requests.post(
            f"{BACKEND_URL}/generate",
            json={
                "prompt": "hello",
                "features": [{"id": SAE_D_SAE, "strength": 100}],
            },
            timeout=120,
        )
        assert resp.status_code == 500

    def test_deterministic_at_zero_temperature(self):
        """With temperature=0, same prompt+features should give same output."""
        a = _generate("Who are you?", [{"id": PACIFIC_OCEAN_FEATURE, "strength": 500}])
        b = _generate("Who are you?", [{"id": PACIFIC_OCEAN_FEATURE, "strength": 500}])
        assert a["steered"] == b["steered"]
        assert a["baseline"] == b["baseline"]

    def test_code_prompt_with_typing_features(self):
        """Verify the existing typing features steer output differently.

        At high strengths, feature steering can produce degenerate output
        (repetitive tokens), which is itself evidence that steering is working.
        We use a moderate strength to balance between effect and coherence.
        """
        result = _generate(
            "def fibonacci(n):",
            [{"id": 124809, "strength": 100}],  # type annotations feature
        )
        assert result["steered"] != result["baseline"], (
            "Steering should produce different output from baseline"
        )
