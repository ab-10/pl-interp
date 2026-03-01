"""Tests for the decode-only steering hook. Uses plain torch tensors, no actual model needed."""

import torch

from experiments.steering.hook import make_steering_hook


def _make_direction(hidden_dim: int = 4096) -> torch.Tensor:
    """Create a normalized random direction vector."""
    d = torch.randn(hidden_dim)
    return d / d.norm()


def test_alpha_zero_decode_noop():
    """At alpha=0, decode step output[0] is unchanged."""
    direction = _make_direction()
    hook = make_steering_hook(direction, alpha=0.0)

    hidden = torch.randn(1, 1, 4096)
    output = (hidden.clone(), None, None)

    result = hook(None, None, output)

    assert torch.equal(result[0], hidden)


def test_alpha_nonzero_decode():
    """At alpha=3.0, decode step: output[0] == original + 3.0 * direction."""
    direction = _make_direction()
    hook = make_steering_hook(direction, alpha=3.0)

    hidden = torch.randn(1, 1, 4096)
    output = (hidden.clone(), None, None)

    result = hook(None, None, output)

    expected = hidden + 3.0 * direction
    assert torch.allclose(result[0], expected, atol=1e-6)


def test_prefill_unchanged():
    """Prefill step (shape[1] > 1): output[0] unchanged regardless of alpha."""
    direction = _make_direction()
    hook = make_steering_hook(direction, alpha=3.0)

    hidden = torch.randn(1, 10, 4096)
    output = (hidden.clone(), None, None)

    result = hook(None, None, output)

    assert torch.equal(result[0], hidden)


def test_output_tail_unchanged_decode():
    """output[1:] always unchanged during decode step."""
    direction = _make_direction()
    hook = make_steering_hook(direction, alpha=3.0)

    hidden = torch.randn(1, 1, 4096)
    attn_weights = torch.randn(1, 8, 1, 64)
    kv_cache = torch.randn(2, 1, 8, 64)
    output = (hidden.clone(), attn_weights, kv_cache)

    result = hook(None, None, output)

    assert result[1] is attn_weights
    assert result[2] is kv_cache


def test_output_tail_unchanged_prefill():
    """output[1:] always unchanged during prefill step."""
    direction = _make_direction()
    hook = make_steering_hook(direction, alpha=5.0)

    hidden = torch.randn(1, 10, 4096)
    attn_weights = torch.randn(1, 8, 10, 64)
    kv_cache = torch.randn(2, 1, 8, 64)
    output = (hidden, attn_weights, kv_cache)

    result = hook(None, None, output)

    assert result[1] is attn_weights
    assert result[2] is kv_cache


def test_return_type_is_tuple():
    """Return type is always a tuple."""
    direction = _make_direction()

    # Decode step with modification
    hook = make_steering_hook(direction, alpha=3.0)
    output_decode = (torch.randn(1, 1, 4096), None, None)
    result_decode = hook(None, None, output_decode)
    assert isinstance(result_decode, tuple)

    # Prefill step (no modification)
    output_prefill = (torch.randn(1, 10, 4096), None, None)
    result_prefill = hook(None, None, output_prefill)
    assert isinstance(result_prefill, tuple)

    # alpha=0 (no modification)
    hook_noop = make_steering_hook(direction, alpha=0.0)
    output_noop = (torch.randn(1, 1, 4096), None, None)
    result_noop = hook_noop(None, None, output_noop)
    assert isinstance(result_noop, tuple)
