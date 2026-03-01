"""Decode-only steering hook for MistralDecoderLayer. Injects a direction vector during decode steps only."""

import torch


def make_steering_hook(
    direction: torch.Tensor,
    alpha: float,
) -> callable:
    """Create a steering hook that adds alpha * direction to hidden states during decode steps.

    The hook only fires when hidden_states.shape[1] == 1 (single-token decode step),
    not during prompt prefill (shape[1] > 1). At alpha=0, this is an exact no-op.

    Args:
        direction: Steering direction vector, shape (hidden_dim,).
        alpha: Scaling factor. 0.0 means no modification.
    """

    def hook(
        module: torch.nn.Module,
        input: tuple,
        output,
    ):
        # MistralDecoderLayer output varies by transformers version:
        #   - Newer (>=4.57): bare Tensor (hidden_states)
        #   - Older: tuple (hidden_states, attn_weights, present_key_value)
        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output

        if hidden_states.shape[1] == 1 and alpha != 0.0:
            # Decode step: inject steering direction
            hidden_states = hidden_states + alpha * direction
            if is_tuple:
                return (hidden_states,) + output[1:]
            return hidden_states

        return output

    return hook


def attach_steering_hook(
    model: torch.nn.Module,
    layer_idx: int,
    direction: torch.Tensor,
    alpha: float,
):
    """Attach a decode-only steering hook to a specific decoder layer.

    Args:
        model: HuggingFace Mistral model.
        layer_idx: Decoder layer index (e.g., 16).
        direction: Steering direction vector.
        alpha: Scaling factor.

    Returns:
        Hook handle — caller can call handle.remove() to detach.
    """
    hook_fn = make_steering_hook(direction, alpha)
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    return handle
