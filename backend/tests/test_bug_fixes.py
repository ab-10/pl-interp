"""Tests reproducing bugs documented in BUGS.md.

Bug 1: encode(mean(activations)) instead of mean(encode(activations))
Bug 2: Last-token-only activation collection
Bug 3: Steering applied during prompt encoding, skipped during generation
Bug 4: Single-token prompts receive no steering
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import pytest

from .conftest import SAE_D_SAE

# ---------------------------------------------------------------------------
# Helpers to import script modules with numeric prefixes (e.g. 01_*.py)
# ---------------------------------------------------------------------------

# Stub additional dependencies needed by scripts (torch/transformers
# are already stubbed in conftest.py).
for _mod in (
    "numpy", "numpy.core", "numpy.core.multiarray",
    "datasets",
    "sae_lens",
    "transformer_lens",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# tqdm must pass through its first argument so for-loops over tqdm(iterable)
# actually iterate.  A plain MagicMock would yield no items.
if "tqdm" not in sys.modules or isinstance(sys.modules["tqdm"], MagicMock):
    _tqdm_passthrough = lambda iterable, *a, **kw: iterable  # noqa: E731
    _tqdm_mod = MagicMock()
    _tqdm_mod.tqdm = _tqdm_passthrough
    # "from tqdm import tqdm" resolves to _tqdm_mod.tqdm
    sys.modules["tqdm"] = _tqdm_mod
if "tqdm.auto" not in sys.modules or isinstance(sys.modules["tqdm.auto"], MagicMock):
    sys.modules["tqdm.auto"] = sys.modules["tqdm"]

_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")


def _import_script(filename: str, module_name: str | None = None):
    """Import a script file by path, even if it has a numeric prefix."""
    path = os.path.join(_SCRIPTS_DIR, filename)
    if module_name is None:
        module_name = filename.removesuffix(".py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Bug 3 & 4: Steering hook behaviour (backend/server.py)
#
# The server uses register_forward_hook to attach a decode-only steering
# hook via make_steering_hook.  These tests verify through the server
# endpoint that the hook is correctly registered when features are active.
# ---------------------------------------------------------------------------


class TestBug3SteeringHookDirection:
    """Bug 3: Steering should apply during generation tokens, not prompt encoding.

    The server uses register_forward_hook to attach a decode-only steering
    hook via make_steering_hook.
    """

    def test_hook_is_registered_for_steered_generation(self, client, mock_model):
        """When features are active, a forward hook should be registered."""
        mock_model._test_layer.register_forward_hook.reset_mock()
        resp = client.post(
            "/generate",
            json={"prompt": "hello", "features": [{"id": 100, "strength": 5.0}]},
        )
        assert resp.status_code == 200
        mock_model._test_layer.register_forward_hook.assert_called_once()

    def test_no_hook_when_no_features(self, client, mock_model):
        """Without active features, no hook should be registered."""
        mock_model._test_layer.register_forward_hook.reset_mock()
        resp = client.post(
            "/generate",
            json={"prompt": "hello", "features": []},
        )
        assert resp.status_code == 200
        mock_model._test_layer.register_forward_hook.assert_not_called()

    def test_hook_is_removed_after_generation(self, client, mock_model):
        """The hook handle should be removed after steered generation."""
        mock_model._test_hook_handle.remove.reset_mock()
        resp = client.post(
            "/generate",
            json={"prompt": "hello", "features": [{"id": 100, "strength": 5.0}]},
        )
        assert resp.status_code == 200
        mock_model._test_hook_handle.remove.assert_called_once()


class TestBug4SingleTokenPrompt:
    """Bug 4: Verify that single-token prompts don't crash or skip steering."""

    def test_single_token_prompt_succeeds(self, client, mock_model):
        """A single-token prompt with steering should succeed."""
        resp = client.post(
            "/generate",
            json={"prompt": "x", "features": [{"id": 100, "strength": 5.0}]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "baseline" in data
        assert "steered" in data


# ---------------------------------------------------------------------------
# Bug 1: encode(mean(x)) != mean(encode(x)) for nonlinear SAE encoder
# ---------------------------------------------------------------------------


class TestBug1EncodeOrder:
    """Bug 1: The SAE encoder is nonlinear (linear + top-k).  Scripts
    mean-pool residual activations BEFORE encoding, which dilutes sparse
    features.  The correct approach is to encode per-token then aggregate.
    """

    def test_explore_trained_sae_encodes_per_token(self):
        """explore_trained_sae.collect_activations should call sae.encode()
        on per-token activations (shape [seq_len, d_model]), not on the
        mean-pooled vector (shape [1, d_model]).
        """
        mod = _import_script("explore_trained_sae.py")
        collect_activations = mod.collect_activations

        seq_len = 10
        d_model = 64

        # Build a mock model whose run_with_cache returns known activations
        mock_model = MagicMock()
        tokens_tensor = MagicMock()
        tokens_tensor.shape = [1, seq_len]
        mock_model.to_tokens.return_value = tokens_tensor

        # The activations tensor returned by the cache
        acts_tensor = MagicMock()
        acts_tensor.shape = [1, seq_len, d_model]

        # Track what .mean() is called with — this is the mean-pooled result
        mean_result = MagicMock(name="mean_pooled_acts")
        mean_result.shape = [1, d_model]
        mean_result.to.return_value = mean_result
        acts_tensor.mean.return_value = mean_result

        cache = MagicMock()
        cache.__getitem__ = lambda self, key: acts_tensor
        mock_model.run_with_cache.return_value = (None, cache)

        # Mock SAE
        mock_sae = MagicMock()
        mock_sae.cfg.d_sae = 1000
        encode_result = MagicMock()
        encode_result.squeeze.return_value = MagicMock()
        encode_result.squeeze.return_value.cpu.return_value = (
            encode_result.squeeze.return_value
        )
        mock_sae.encode.return_value = encode_result

        collect_activations(mock_model, mock_sae, "hook_point", "test text", "cpu")

        # The function should encode per-token activations.
        # The buggy code calls acts.mean(dim=1) first, then passes that
        # mean_result to sae.encode().  Correct code encodes all tokens first.
        encode_arg = mock_sae.encode.call_args[0][0]
        assert encode_arg is not mean_result, (
            "Bug 1: sae.encode() was called on the mean-pooled activations "
            "instead of per-token activations.  The SAE encoder is nonlinear, "
            "so encode(mean(x)) != mean(encode(x))."
        )

    def test_prompt_and_observe_encodes_per_token(self):
        """05_prompt_and_observe.collect_top_features should encode per-token
        activations, not the mean-pooled residual.
        """
        mod = _import_script("05_prompt_and_observe.py", "prompt_and_observe")
        collect_top_features = mod.collect_top_features

        seq_len = 15
        d_model = 64

        mock_model = MagicMock()
        tokens = MagicMock()
        tokens.shape = [1, seq_len]
        mock_model.to_tokens.return_value = tokens

        # model.generate returns output tokens
        output_tokens = MagicMock()
        output_tokens.__getitem__ = lambda self, idx: MagicMock()
        mock_model.generate.return_value = output_tokens
        mock_model.tokenizer.decode.return_value = "generated text"

        # Set up cache activations
        acts_tensor = MagicMock()
        acts_tensor.shape = [1, seq_len, d_model]

        # acts[0] is the per-token activations
        per_token = MagicMock(name="per_token_acts")
        per_token.shape = [seq_len, d_model]
        per_token.float.return_value = per_token

        # acts[0].mean(dim=0) produces a mean vector
        mean_vector = MagicMock(name="mean_vector")
        mean_vector.shape = [d_model]
        mean_vector.float.return_value = mean_vector
        unsqueezed_mean = MagicMock(name="unsqueezed_mean")
        unsqueezed_mean.shape = [1, d_model]
        mean_vector.unsqueeze.return_value = unsqueezed_mean
        per_token.mean.return_value = mean_vector

        # acts[0, -1, :] is the last token
        last_token = MagicMock(name="last_token")
        last_token.shape = [d_model]
        last_token.float.return_value = last_token
        last_token.unsqueeze.return_value = MagicMock(name="last_unsqueezed")

        def getitem(idx):
            if idx == 0:
                return per_token
            if isinstance(idx, tuple) and len(idx) == 3:
                return last_token
            return MagicMock()

        acts_tensor.__getitem__ = lambda self, idx: getitem(idx)

        cache = MagicMock()
        cache.__getitem__ = lambda self, key: acts_tensor
        mock_model.run_with_cache.return_value = (None, cache)

        # Mock SAE
        mock_sae = MagicMock()
        mock_sae.cfg.d_sae = 1000

        # Make encode return something with topk support
        fake_feats = MagicMock()
        fake_feats.squeeze.return_value = fake_feats
        top_vals = MagicMock()
        top_vals.cpu.return_value = []
        top_idxs = MagicMock()
        top_idxs.cpu.return_value = []

        import torch as torch_mock

        torch_mock.topk = MagicMock(return_value=(top_vals, top_idxs))
        fake_feats.topk = MagicMock(return_value=(top_vals, top_idxs))

        mock_sae.encode.return_value = fake_feats

        result = collect_top_features(
            mock_model,
            mock_sae,
            "hook_point",
            "test prompt",
            max_new_tokens=10,
            temperature=0.3,
            top_k=5,
        )

        # Verify: the FIRST call to sae.encode should be with per-token
        # activations, not the mean-pooled vector.
        first_encode_arg = mock_sae.encode.call_args_list[0][0][0]
        assert first_encode_arg is not unsqueezed_mean, (
            "Bug 1: sae.encode() was called on mean-pooled activations "
            "instead of per-token activations.  "
            "encode(mean(x)) != mean(encode(x)) for nonlinear encoders."
        )


# ---------------------------------------------------------------------------
# Bug 2: Last-token-only collection discards most of the signal
# ---------------------------------------------------------------------------


class TestBug2LastTokenOnly:
    """Bug 2: 01_collect_activations.py extracts features from only the
    last token position, discarding activations from positions 0..seq_len-2.
    """

    def test_collect_activations_uses_all_positions(self):
        """All token positions should be encoded through the SAE, not just
        the last one.
        """
        mod = _import_script("01_collect_activations.py", "collect_activations_script")
        collect_activations = mod.collect_activations

        seq_len = 20
        d_model = 64

        mock_model = MagicMock()
        tokens = MagicMock()
        tokens.shape = [1, seq_len]
        mock_model.to_tokens.return_value = tokens

        # Create activation tensor
        acts_tensor = MagicMock()
        acts_tensor.shape = [1, seq_len, d_model]

        # acts[0] is all positions (shape [seq_len, d_model])
        all_positions = MagicMock(name="all_positions")
        all_positions.shape = [seq_len, d_model]
        all_positions.float.return_value = all_positions

        # acts[0, -1, :] is last position only (shape [d_model])
        last_position = MagicMock(name="last_position")
        last_position.shape = [d_model]
        last_position.float.return_value = last_position
        last_unsqueezed = MagicMock(name="last_unsqueezed")
        last_unsqueezed.shape = [1, d_model]
        last_position.unsqueeze.return_value = last_unsqueezed

        def getitem(idx):
            if idx == 0:
                return all_positions
            if isinstance(idx, tuple):
                # (0, -1, :) or similar — last token extraction
                return last_position
            return MagicMock()

        acts_tensor.__getitem__ = lambda self, idx: getitem(idx)

        cache = MagicMock()
        cache.__getitem__ = lambda self, key: acts_tensor
        mock_model.run_with_cache.return_value = (None, cache)

        # Mock SAE
        mock_sae = MagicMock()
        mock_sae.cfg.d_sae = 1000

        # encode returns feature activations
        feat_acts = MagicMock()
        feat_acts.squeeze.return_value = feat_acts
        feat_acts.shape = [1000]

        import torch as torch_mock

        top_vals = MagicMock()
        top_vals.cpu.return_value = []
        top_idxs = MagicMock()
        top_idxs.cpu.return_value = []
        torch_mock.topk = MagicMock(return_value=(top_vals, top_idxs))
        feat_acts.topk = MagicMock(return_value=(top_vals, top_idxs))

        mock_sae.encode.return_value = feat_acts

        prompt_info = [
            {"id": "test_1", "source": "test", "text": "hello world"},
        ]

        results = collect_activations(
            mock_model, mock_sae, "hook_point", prompt_info, top_k=5, label="test"
        )

        # Check what was passed to sae.encode().
        # Bug: the current code passes last_position.float().unsqueeze(0),
        # which is a [1, d_model] tensor from just the last token.
        # Correct: should pass all_positions.float(), shape [seq_len, d_model].
        encode_arg = mock_sae.encode.call_args[0][0]
        assert encode_arg is not last_unsqueezed, (
            "Bug 2: sae.encode() was called with only the last token's "
            "activations (shape [1, d_model]) instead of all token "
            "positions (shape [seq_len, d_model]).  This discards "
            f"{seq_len - 1} of {seq_len} positions worth of signal."
        )
