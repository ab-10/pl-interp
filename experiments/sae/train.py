"""Stratified SAE training loop. Trains a TopK sparse autoencoder on pass/fail-balanced activations."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterator

import os

import numpy as np
import torch

from experiments import config

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


def _wandb_enabled() -> bool:
    if _wandb is None:
        return False
    return os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1")
from experiments.sae.model import TopKSAE, sae_loss
from experiments.storage.activation_store import ActivationReader
from experiments.storage.schema import GenerationRecord, read_records


# ---------------------------------------------------------------------------
# Stratified activation loader
# ---------------------------------------------------------------------------


class StratifiedActivationLoader:
    """Yields 50/50 pass/fail balanced activation batches from mmap shard files.

    Each batch has shape (batch_size, 4096) in float32. Tokens are drawn
    equally from passed and failed generation records, shuffled within each
    pool at the start of every iteration.
    """

    def __init__(
        self,
        records: list[GenerationRecord],
        activation_dir: Path,
        batch_size: int,
        token_budget: int,
    ) -> None:
        self.batch_size = batch_size
        self.token_budget = token_budget
        self.activation_dir = activation_dir

        # Split records into pass / fail pools
        pass_records = [r for r in records if r.passed]
        fail_records = [r for r in records if not r.passed]

        # Build per-pool token index: list of (shard_path, row_offset_within_shard)
        self.pass_pool = self._build_pool(pass_records)
        self.fail_pool = self._build_pool(fail_records)

        # Cache open readers keyed by shard path
        self._readers: dict[Path, ActivationReader] = {}

    # ------------------------------------------------------------------

    def _build_pool(
        self,
        records: list[GenerationRecord],
    ) -> list[tuple[Path, int]]:
        """Expand records into a flat list of (shard_path, row_index) entries."""
        pool: list[tuple[Path, int]] = []
        for rec in records:
            shard_path = Path(rec.activation_file)
            # Make path absolute relative to activation_dir if needed
            if not shard_path.is_absolute():
                shard_path = self.activation_dir / shard_path
            for row in range(rec.activation_offset, rec.activation_offset + rec.activation_length):
                pool.append((shard_path, row))
        return pool

    def _get_reader(self, shard_path: Path) -> ActivationReader:
        if shard_path not in self._readers:
            self._readers[shard_path] = ActivationReader(shard_path)
        return self._readers[shard_path]

    def _read_row(self, shard_path: Path, row: int) -> np.ndarray:
        reader = self._get_reader(shard_path)
        return reader.read(row, 1)  # shape (1, 4096), float16

    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[torch.Tensor]:
        # Shuffle both pools independently
        pass_indices = list(range(len(self.pass_pool)))
        fail_indices = list(range(len(self.fail_pool)))
        random.shuffle(pass_indices)
        random.shuffle(fail_indices)

        half = self.batch_size // 2
        pass_ptr = 0
        fail_ptr = 0
        tokens_yielded = 0

        while tokens_yielded < self.token_budget:
            # Wrap around pools if exhausted (with reshuffle)
            if pass_ptr + half > len(pass_indices):
                random.shuffle(pass_indices)
                pass_ptr = 0
            if fail_ptr + half > len(fail_indices):
                random.shuffle(fail_indices)
                fail_ptr = 0

            rows: list[np.ndarray] = []

            # Collect pass half
            for i in range(half):
                idx = pass_indices[pass_ptr + i]
                shard_path, row = self.pass_pool[idx]
                rows.append(self._read_row(shard_path, row))
            pass_ptr += half

            # Collect fail half
            for i in range(half):
                idx = fail_indices[fail_ptr + i]
                shard_path, row = self.fail_pool[idx]
                rows.append(self._read_row(shard_path, row))
            fail_ptr += half

            # Stack into (batch_size, 4096) float32 tensor
            batch_np = np.concatenate(rows, axis=0).astype(np.float32)
            batch = torch.from_numpy(batch_np)

            tokens_yielded += batch.shape[0]
            yield batch


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _cosine_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup for warmup_steps, then cosine decay to 0."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_sae(
    records: list[GenerationRecord],
    activation_dir: Path,
    output_dir: Path,
    batch_size: int = 4096,
    lr: float = 3e-4,
    device: str = "cuda",
    token_budget: int = config.SAE_TRAINING_TOKENS,
    d_model: int = config.MODEL_HIDDEN_DIM,
    d_sae: int = config.SAE_NUM_FEATURES,
    k: int = config.SAE_K,
) -> Path:
    """Train a TopK SAE on stratified pass/fail activations.

    Returns the path to the saved checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Model ---
    sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=k).to(device)

    # --- Optimizer & schedule ---
    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)
    total_steps = token_budget // batch_size
    warmup_steps = max(1, int(0.05 * total_steps))
    scheduler = _cosine_warmup_schedule(optimizer, warmup_steps, total_steps)

    # --- Data ---
    loader = StratifiedActivationLoader(records, activation_dir, batch_size, token_budget)

    # --- Dead feature tracking ---
    dead_steps_threshold = 500
    last_fired = torch.zeros(d_sae, dtype=torch.long, device=device)  # step when last fired
    resample_interval = 500

    # --- Logging ---
    log_path = output_dir / "training_log.jsonl"
    log_interval = 50

    # --- wandb ---
    use_wandb = _wandb_enabled()
    if use_wandb:
        _wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name="sae-training",
            config={
                "d_model": d_model, "d_sae": d_sae, "k": k,
                "lr": lr, "batch_size": batch_size, "token_budget": token_budget,
                "warmup_steps": warmup_steps, "total_steps": total_steps,
                "pass_pool": len(loader.pass_pool), "fail_pool": len(loader.fail_pool),
            },
        )
        print("  wandb: logging enabled")

    print(f"Training TopKSAE: d_model={d_model}, d_sae={d_sae}, k={k}")
    print(f"  tokens={token_budget}, batch_size={batch_size}, steps~={total_steps}")
    print(f"  lr={lr}, warmup={warmup_steps}, device={device}")
    print(f"  pass_pool={len(loader.pass_pool)} tokens, fail_pool={len(loader.fail_pool)} tokens")

    sae.train()
    step = 0

    for batch in loader:
        batch = batch.to(device)

        # Forward
        x_hat, topk_latents, info = sae(batch)

        # Dead feature mask: features not fired in last dead_steps_threshold steps
        dead_mask = (step - last_fired) >= dead_steps_threshold if step >= dead_steps_threshold else None

        # Loss
        loss_dict = sae_loss(batch, x_hat, topk_latents, sae, dead_mask=dead_mask)
        loss = loss_dict["loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Normalize decoder columns
        sae.normalize_decoder()

        # Update feature firing tracker
        topk_indices = info["topk_indices"]  # shape (batch_size, k)
        fired = torch.unique(topk_indices.flatten())
        last_fired[fired] = step

        # Dead feature resampling
        if step > 0 and step % resample_interval == 0:
            _resample_dead_features(
                sae, optimizer, batch, x_hat, last_fired, step,
                dead_steps_threshold, d_model, d_sae, device,
            )

        # Logging
        if step % log_interval == 0:
            with torch.no_grad():
                # Variance explained: 1 - MSE / Var(x)
                x_var = batch.var().item()
                var_explained = 1.0 - (loss_dict["mse"].item() / max(x_var, 1e-8))
                dead_count = int((step - last_fired >= dead_steps_threshold).sum().item()) if step >= dead_steps_threshold else 0

            log_entry = {
                "step": step,
                "loss": loss_dict["loss"].item(),
                "mse": loss_dict["mse"].item(),
                "aux": loss_dict["aux"].item(),
                "l0": float(k),
                "dead_features": dead_count,
                "var_explained": var_explained,
                "lr": scheduler.get_last_lr()[0],
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if use_wandb:
                _wandb.log(log_entry, step=step)

            if step % (log_interval * 5) == 0:
                print(
                    f"  step {step}/{total_steps}  "
                    f"loss={log_entry['loss']:.4f}  mse={log_entry['mse']:.4f}  "
                    f"aux={log_entry['aux']:.4f}  dead={log_entry['dead_features']}  "
                    f"var_expl={log_entry['var_explained']:.3f}  lr={log_entry['lr']:.2e}"
                )

        step += 1

    # --- Save checkpoint ---
    checkpoint_path = output_dir / "sae_checkpoint.pt"
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "config": {"d_model": d_model, "d_sae": d_sae, "k": k},
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}  (final step: {step})")

    if use_wandb:
        _wandb.finish()

    return checkpoint_path


# ---------------------------------------------------------------------------
# Dead feature resampling
# ---------------------------------------------------------------------------


def _resample_dead_features(
    sae: TopKSAE,
    optimizer: torch.optim.AdamW,
    batch: torch.Tensor,
    x_hat: torch.Tensor,
    last_fired: torch.Tensor,
    step: int,
    threshold: int,
    d_model: int,
    d_sae: int,
    device: str,
) -> None:
    """Resample dead encoder/decoder rows from high-loss inputs."""
    dead_mask = (step - last_fired) >= threshold
    dead_indices = dead_mask.nonzero(as_tuple=True)[0]

    if len(dead_indices) == 0:
        return

    # Find high-loss inputs
    with torch.no_grad():
        per_token_mse = (batch - x_hat).pow(2).mean(dim=-1)  # (batch_size,)
        # Pick tokens with loss above median
        median_loss = per_token_mse.median()
        high_loss_mask = per_token_mse > median_loss
        high_loss_inputs = batch[high_loss_mask]

    if high_loss_inputs.shape[0] == 0:
        return

    n_dead = len(dead_indices)

    # Sample from high-loss inputs (with replacement if needed)
    sample_indices = torch.randint(0, high_loss_inputs.shape[0], (n_dead,), device=device)
    sampled = high_loss_inputs[sample_indices]  # (n_dead, d_model)

    # Normalize to unit vectors
    sampled = sampled / (sampled.norm(dim=-1, keepdim=True) + 1e-8)

    # Re-init encoder rows and decoder columns
    with torch.no_grad():
        sae.W_enc.data[:, dead_indices] = sampled.T  # W_enc: (d_model, d_sae)
        sae.W_dec.data[dead_indices, :] = sampled     # W_dec: (d_sae, d_model)
        sae.b_enc.data[dead_indices] = 0.0

    # Reset Adam state for the resampled parameters
    for param in [sae.W_enc, sae.W_dec, sae.b_enc]:
        state = optimizer.state.get(param, {})
        if "exp_avg" in state:
            if param is sae.W_enc:
                state["exp_avg"][:, dead_indices] = 0.0
                state["exp_avg_sq"][:, dead_indices] = 0.0
            elif param is sae.W_dec:
                state["exp_avg"][dead_indices, :] = 0.0
                state["exp_avg_sq"][dead_indices, :] = 0.0
            elif param is sae.b_enc:
                state["exp_avg"][dead_indices] = 0.0
                state["exp_avg_sq"][dead_indices] = 0.0

    # Mark resampled features as alive
    last_fired[dead_indices] = step

    if _wandb_enabled() and _wandb.run is not None:
        _wandb.log({"dead_resampled": n_dead}, step=step)

    print(f"  [resample] step {step}: resampled {n_dead} dead features")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_records_from_dir(generations_dir: Path) -> list[GenerationRecord]:
    """Load all GenerationRecords from .jsonl files in a directory."""
    records: list[GenerationRecord] = []
    for jsonl_path in sorted(generations_dir.glob("*.jsonl")):
        records.extend(read_records(jsonl_path))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TopK SAE on stratified activations")
    parser.add_argument("--generations-dir", type=Path, default=config.GENERATIONS_DIR)
    parser.add_argument("--activations-dir", type=Path, default=config.ACTIVATIONS_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.SAE_DIR)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading generation records from {args.generations_dir}")
    records = _load_records_from_dir(args.generations_dir)
    print(f"  {len(records)} records loaded")

    n_pass = sum(1 for r in records if r.passed)
    n_fail = len(records) - n_pass
    print(f"  pass={n_pass}, fail={n_fail}")

    checkpoint_path = train_sae(
        records=records,
        activation_dir=args.activations_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    print(f"\nTraining complete. Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
