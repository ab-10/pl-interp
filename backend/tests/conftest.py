"""Shared fixtures for backend tests."""

import os
import sys
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest


BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

SAE_D_SAE = 32_768  # dictionary size of the SAE


# ---------------------------------------------------------------------------
# Stub heavy ML libraries so we can import backend.server without a GPU.
# ---------------------------------------------------------------------------
def _install_ml_stubs():
    """Insert lightweight stubs for torch / transformers / experiments so
    that ``import backend.server`` succeeds in a CPU-only environment."""
    for mod_name in (
        "torch",
        "torch.cuda",
        "torch.nn",
        "transformers",
        "experiments",
        "experiments.sae",
        "experiments.sae.model",
        "experiments.steering",
        "experiments.steering.hook",
    ):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()


_install_ml_stubs()

from fastapi.testclient import TestClient  # noqa: E402


# ---------------------------------------------------------------------------
# Mocked TestClient for unit tests (no GPU required)
# ---------------------------------------------------------------------------
@pytest.fixture()
def mock_model():
    """Return a lightweight mock that behaves like AutoModelForCausalLM."""
    m = MagicMock()
    # model.generate returns a tensor-like with indexing support
    output_ids = MagicMock()
    output_ids.__getitem__ = lambda self, idx: MagicMock()
    m.generate.return_value = output_ids
    # model.parameters() for device detection
    param = MagicMock()
    param.device = "cpu"
    m.parameters.return_value = iter([param])
    # model.model.layers[i].register_forward_hook
    layer = MagicMock()
    handle = MagicMock()
    layer.register_forward_hook.return_value = handle
    # Use a MagicMock for __getitem__ so call tracking works
    layers_mock = MagicMock()
    layers_mock.__getitem__ = MagicMock(return_value=layer)
    m.model.layers = layers_mock
    # Expose the layer mock for easy test access
    m._test_layer = layer
    m._test_hook_handle = handle
    return m


@pytest.fixture()
def mock_tokenizer():
    """Return a lightweight mock that behaves like AutoTokenizer."""
    t = MagicMock()
    input_ids = MagicMock()
    input_ids.shape = [1, 5]
    # tokenizer(prompt, return_tensors="pt") returns a BatchEncoding-like object
    # that supports .to(device) and ["input_ids"] access
    tok_output = MagicMock()
    tok_output.__getitem__ = lambda self, key: input_ids if key == "input_ids" else MagicMock()
    tok_output.to.return_value = tok_output
    t.return_value = tok_output
    t.decode.return_value = "def fibonacci(n):\n    if n <= 1:\n        return n"
    t.eos_token_id = 2
    return t


def _make_mock_sae():
    """Create a mock SAE matching custom TopKSAE interface."""
    import torch as torch_mock

    s = MagicMock()
    # W_dec shape: (d_sae, d_model)
    w_dec = MagicMock()
    w_dec.shape = [SAE_D_SAE, 4096]
    vec = MagicMock()
    vec.to = MagicMock(return_value=vec)
    w_dec.__getitem__ = lambda self, idx: vec
    s.W_dec = w_dec
    return s


@pytest.fixture()
def mock_sae():
    """Return a lightweight mock SAE."""
    return _make_mock_sae()


@pytest.fixture()
def client(mock_model, mock_tokenizer, mock_sae):
    """FastAPI TestClient with mocked model, tokenizer & SAE — lifespan skipped."""
    import backend.server as srv

    # Inject mocks into module-level globals
    srv.model = mock_model
    srv.tokenizer = mock_tokenizer
    srv.sae = mock_sae
    srv.feature_labels = {"100": "test feature", "200": "another feature"}

    # Replace the lifespan with a no-op so the TestClient doesn't try to
    # load the real model (which needs CUDA).
    @asynccontextmanager
    async def _noop_lifespan(app):
        yield

    original_lifespan = srv.app.router.lifespan_context
    srv.app.router.lifespan_context = _noop_lifespan

    with TestClient(srv.app, raise_server_exceptions=False) as c:
        yield c

    # Restore
    srv.app.router.lifespan_context = original_lifespan
    srv.model = None
    srv.tokenizer = None
    srv.sae = None
    srv.feature_labels = {}
