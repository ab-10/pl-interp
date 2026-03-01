"""TopK Sparse Autoencoder. Encodes residual-stream activations into sparse latent codes with exactly K active features."""

import math

import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    """Sparse autoencoder with TopK activation sparsity.

    Encodes d_model activations into d_sae latent features, keeping only
    the top-k activations per input. The decoder rows W_dec[i] give the
    (d_model,) steering direction for feature i.
    """

    def __init__(self, d_model: int, d_sae: int, k: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_pre = nn.Parameter(torch.zeros(d_model))

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming uniform for encoder, unit-norm random rows for decoder."""
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))

        # Decoder rows: random unit-norm vectors
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Encode, apply TopK sparsity, decode.

        Args:
            x: Input activations, shape (batch, d_model).

        Returns:
            x_hat: Reconstructed activations, shape (batch, d_model).
            topk_latents: Sparse latent codes, shape (batch, d_sae).
            info: Dict with "topk_indices" (batch, k) and "topk_values" (batch, k).
        """
        x_centered = x - self.b_pre
        latents = x_centered @ self.W_enc + self.b_enc

        topk_values, topk_indices = torch.topk(latents, self.k, dim=-1)
        topk_latents = torch.zeros_like(latents).scatter_(
            1, topk_indices, topk_values
        )

        x_hat = topk_latents @ self.W_dec + self.b_pre

        return x_hat, topk_latents, {
            "topk_indices": topk_indices,
            "topk_values": topk_values,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without TopK — returns full pre-activation latents for analysis.

        Args:
            x: Input activations, shape (batch, d_model).

        Returns:
            Full latent activations, shape (batch, d_sae).
        """
        x_centered = x - self.b_pre
        return x_centered @ self.W_enc + self.b_enc

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Normalize each row of W_dec to unit norm."""
        self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True)


def sae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    topk_latents: torch.Tensor,
    sae: TopKSAE,
    dead_mask: torch.Tensor | None = None,
    aux_coeff: float = 1 / 32,
) -> dict[str, torch.Tensor]:
    """Compute SAE training loss: MSE reconstruction + auxiliary dead feature loss.

    Args:
        x: Original activations, shape (batch, d_model).
        x_hat: Reconstructed activations, shape (batch, d_model).
        topk_latents: Sparse latent codes, shape (batch, d_sae).
        sae: The TopKSAE model (used for dead feature re-encoding).
        dead_mask: Boolean mask of shape (d_sae,) — True for dead features.
        aux_coeff: Weight for auxiliary dead feature loss.

    Returns:
        Dict with "loss", "mse", and "aux" tensors.
    """
    mse = (x - x_hat).pow(2).mean()

    aux = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    if dead_mask is not None and dead_mask.any():
        # Re-encode residual through dead features only
        residual = x - x_hat
        residual_centered = residual - sae.b_pre
        residual_latents = residual_centered @ sae.W_enc + sae.b_enc

        # Zero out alive features, keep only dead
        dead_latents = residual_latents * dead_mask.float().unsqueeze(0)

        # Decode from dead features only
        residual_hat = dead_latents @ sae.W_dec + sae.b_pre
        aux = (residual - residual_hat).pow(2).mean()

    loss = mse + aux_coeff * aux

    return {"loss": loss, "mse": mse, "aux": aux}
