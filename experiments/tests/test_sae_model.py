"""Tests for TopK Sparse Autoencoder. Uses tiny dimensions (d_model=64, d_sae=256, K=4)."""

import torch

from experiments.sae.model import TopKSAE, sae_loss

D_MODEL = 64
D_SAE = 256
K = 4
BATCH = 8


def _make_sae() -> TopKSAE:
    """Create a small SAE for testing."""
    return TopKSAE(d_model=D_MODEL, d_sae=D_SAE, k=K)


def test_forward_shapes():
    """Forward pass produces correct output shapes."""
    sae = _make_sae()
    x = torch.randn(BATCH, D_MODEL)

    x_hat, topk_latents, info = sae(x)

    assert x_hat.shape == (BATCH, D_MODEL)
    assert topk_latents.shape == (BATCH, D_SAE)
    assert info["topk_indices"].shape == (BATCH, K)
    assert info["topk_values"].shape == (BATCH, K)


def test_topk_sparsity():
    """Exactly K non-zero values per row in the sparse latent codes."""
    sae = _make_sae()
    x = torch.randn(BATCH, D_MODEL)

    _, topk_latents, _ = sae(x)

    nonzero_per_row = (topk_latents != 0).sum(dim=1)
    assert (nonzero_per_row == K).all()


def test_loss_decreases():
    """Loss decreases over 10 training steps."""
    sae = _make_sae()
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
    x = torch.randn(32, D_MODEL)

    # Record initial loss
    x_hat, topk_latents, _ = sae(x)
    initial_loss = sae_loss(x, x_hat, topk_latents, sae)["loss"].item()

    # Train for 10 steps
    for _ in range(10):
        x_hat, topk_latents, _ = sae(x)
        loss_dict = sae_loss(x, x_hat, topk_latents, sae)
        optimizer.zero_grad()
        loss_dict["loss"].backward()
        optimizer.step()

    # Record final loss
    x_hat, topk_latents, _ = sae(x)
    final_loss = sae_loss(x, x_hat, topk_latents, sae)["loss"].item()

    assert final_loss < initial_loss


def test_weight_shapes():
    """Encoder and decoder weight matrices have correct shapes."""
    sae = _make_sae()

    assert sae.W_enc.shape == (D_MODEL, D_SAE)
    assert sae.W_dec.shape == (D_SAE, D_MODEL)
    assert sae.b_enc.shape == (D_SAE,)
    assert sae.b_pre.shape == (D_MODEL,)


def test_gradient_flow():
    """After backward, encoder and decoder gradients are non-zero."""
    sae = _make_sae()
    x = torch.randn(BATCH, D_MODEL)

    x_hat, topk_latents, _ = sae(x)
    loss_dict = sae_loss(x, x_hat, topk_latents, sae)
    loss_dict["loss"].backward()

    assert sae.W_enc.grad is not None
    assert not torch.all(sae.W_enc.grad == 0)
    assert sae.W_dec.grad is not None
    assert not torch.all(sae.W_dec.grad == 0)


def test_decoder_normalization():
    """After normalize_decoder(), all decoder row norms are approximately 1.0."""
    sae = _make_sae()

    # Perturb decoder weights away from unit norm
    with torch.no_grad():
        sae.W_dec.data *= torch.randn(D_SAE, 1).abs() + 0.5

    sae.normalize_decoder()

    row_norms = sae.W_dec.data.norm(dim=1)
    assert torch.allclose(row_norms, torch.ones(D_SAE), atol=1e-6)


def test_encode_returns_full_latents():
    """encode() returns full pre-TopK latents — not zeroed by TopK."""
    sae = _make_sae()
    x = torch.randn(BATCH, D_MODEL)

    full_latents = sae.encode(x)

    assert full_latents.shape == (BATCH, D_SAE)
    # Full latents should have many more non-zero entries than K per row
    nonzero_per_row = (full_latents != 0).sum(dim=1)
    assert (nonzero_per_row > K).all()
