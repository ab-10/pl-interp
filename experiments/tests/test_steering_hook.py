"""Tests for the decode-only steering hook. Uses plain torch tensors, no actual model needed.

Tests both tuple and bare tensor output formats (transformers version dependent).
"""

import torch

from experiments.steering.hook import make_steering_hook


def _make_direction(hidden_dim: int = 4096) -> torch.Tensor:
    """Create a normalized random direction vector."""
    d = torch.randn(hidden_dim)
    return d / d.norm()


# --- Tuple output format (older transformers) ---

def test_tuple_alpha_zero_decode_noop():
    """Tuple output: alpha=0, decode step — output unchanged."""
    direction = _make_direction()
    hook = make_steering_hook(direction, alpha=0.0)

    hidden = torch.randn(1, 1, 4096)
    output = (hidden.clone(), None, None)

    result = hook(None, None, output)

    assert isinstance(result, tuple)
    assert torch.equal(result[0], hidden)


def test_tuple_alpha_nonzero_decode():
    """Tuple output: alpha=3.0, decode step — modification applied."""
    direction = _make_direction()
    hook = make_steering_hook(direction, alpha=3.0)

    hidden = torch.randn(1, 1, 4096)
    output = (hidden.clone(), None, None)

    result = hook(None, None, output)

    expected = hidden + 3.0 * direction
    assert isinstance(result, tuple)
    assert torch.allclose(result[0], expected, atol=1e-6)


def test_tuple_prefill_unchanged():
    """Tuple output: prefill step (shape[1] > 1) — unchanged."""
    direction = _make_direction()
    hook = make_steering_hook(direction, alpha=3.0)

    hidden = torch.randn(1, 10, 4096)
    output = (hidden.clone(), None, None)

    result = hook(None, None, output)

    assert isinstance(result, tuple)
    assert torch.equal(result[0], hidden)


def test_tuple_tail_unchanged():
    """Tuple output: output[1:] preserved during decode step."""
    direction = _make_direction()
    hook = make_steering_hook(direction, alpha=3.0)

    hidden = torch.randn(1, 1, 4096)
    attn_weights = torch.randn(1, 8, 1, 64)
    kv_cache = torch.randn(2, 1, 8, 64)
    output = (hidden.clone(), attn_weights, kv_cache)

    result = hook(None, None, output)

    assert result[1] is attn_weights
    assert result[2] is kv_cache


# --- Bare tensor output format (newer transformers >=4.57) ---

def test_tensor_alpha_zero_decode_noop():
    """Bare tensor output: alpha=0, decode step — output unchanged."""
    direction = _make_direction()
    hook = make_steering_hook(direction, alpha=0.0)

    hidden = torch.randn(1, 1, 4096)
    result = hook(None, None, hidden.clone())

    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, hidden)


def test_tensor_alpha_nonzero_decode():
    """Bare tensor output: alpha=3.0, decode step — modification applied."""
    direction = _make_direction()
    hook = make_steering_hook(direction, alpha=3.0)

    hidden = torch.randn(1, 1, 4096)
    result = hook(None, None, hidden.clone())

    expected = hidden + 3.0 * direction
    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, expected, atol=1e-6)


def test_tensor_prefill_unchanged():
    """Bare tensor output: prefill step (shape[1] > 1) — unchanged."""
    direction = _make_direction()
    hook = make_steering_hook(direction, alpha=3.0)

    hidden = torch.randn(1, 10, 4096)
    result = hook(None, None, hidden.clone())

    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, hidden)


def test_tensor_returns_tensor_not_tuple():
    """Bare tensor input → bare tensor output (not wrapped in tuple)."""
    direction = _make_direction()
    hook = make_steering_hook(direction, alpha=3.0)

    hidden = torch.randn(1, 1, 4096)
    result = hook(None, None, hidden.clone())

    assert isinstance(result, torch.Tensor)
    assert not isinstance(result, tuple)
