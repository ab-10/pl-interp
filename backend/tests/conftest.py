"""Shared fixtures for backend tests."""

import os
import sys
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Blog-post feature IDs (from Tyler Cosgrove's SAE analysis)
# ---------------------------------------------------------------------------
PACIFIC_OCEAN_FEATURE = 79557
BITTERNESS_FEATURE = 101594
RHYMING_FEATURE = 131062

# Existing verified features in the server registry
REGISTRY_FEATURES = {124809, 6133, 8019, 28468, 95915, 70728}

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

SAE_D_SAE = 131_072  # dictionary size of the SAE


# ---------------------------------------------------------------------------
# Stub heavy ML libraries so we can import backend.server without a GPU.
# ---------------------------------------------------------------------------
def _install_ml_stubs():
    """Insert lightweight stubs for torch / sae_lens / transformer_lens so
    that ``import backend.server`` succeeds in a CPU-only environment."""
    for mod_name in (
        "torch",
        "torch.cuda",
        "sae_lens",
        "transformer_lens",
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
    """Return a lightweight mock that behaves like HookedTransformer."""
    m = MagicMock()
    m.generate.return_value = "def fibonacci(n):\n    if n <= 1:\n        return n"
    # Support context-manager hooks interface
    m.hooks.return_value.__enter__ = lambda s: s
    m.hooks.return_value.__exit__ = MagicMock(return_value=False)
    return m


@pytest.fixture()
def mock_sae():
    """Return a lightweight mock that behaves like sae_lens.SAE."""
    s = MagicMock()
    s.cfg.d_sae = SAE_D_SAE
    s.cfg.metadata = {"hook_name": "blocks.16.hook_mlp_out"}
    # W_dec[idx].detach().clone() chain
    vec = MagicMock()
    vec.detach.return_value.clone.return_value = MagicMock()
    vec.detach.return_value.clone.return_value.to = MagicMock(return_value=vec)
    s.W_dec.__getitem__ = lambda self, idx: vec
    return s


@pytest.fixture()
def client(mock_model, mock_sae):
    """FastAPI TestClient with mocked model & SAE globals — lifespan skipped."""
    import backend.server as srv

    # Inject mocks into module-level globals
    srv.model = mock_model
    srv.sae = mock_sae
    srv.hook_point = "blocks.16.hook_mlp_out"

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
    srv.sae = None
    srv.hook_point = None
