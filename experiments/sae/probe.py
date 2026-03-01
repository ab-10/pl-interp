"""Train a linear probe on SAE feature activations to find the optimal pass/fail
steering direction. Projects the learned weight vector back to hidden-state space
via the SAE decoder matrix.

Usage:
  python -m experiments.sae.probe \
    --sae-checkpoint /scratch/ministral-8b/sae/sae_checkpoint.pt \
    --generations-dir /scratch/ministral-8b/generations \
    --activations-dir /scratch/ministral-8b/activations \
    --output-dir /scratch/ministral-8b/analysis \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from experiments import config
from experiments.sae.model import TopKSAE
from experiments.storage.activation_store import ActivationReader
from experiments.storage.schema import GenerationRecord, read_records

logger = logging.getLogger(__name__)


def _load_sae(checkpoint_path: Path, device: str) -> TopKSAE:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sae = TopKSAE(d_model=cfg["d_model"], d_sae=cfg["d_sae"], k=cfg["k"])
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(device)
    sae.eval()
    return sae


def _load_all_records(generations_dir: Path) -> list[GenerationRecord]:
    records: list[GenerationRecord] = []
    for path in sorted(generations_dir.glob("*.jsonl")):
        records.extend(read_records(path))
    return records


def _get_record_sae_features(
    record: GenerationRecord,
    sae: TopKSAE,
    readers: dict[str, ActivationReader],
    device: str,
    layer: int | None = None,
) -> np.ndarray | None:
    """Compute mean SAE feature activations (TopK sparse) for a single record.

    Uses activation_layers dict from the record. If layer is None, uses the
    first available layer (typically the primary capture layer).

    Returns a (d_sae,) float32 array, or None if activations are missing.
    """
    if not record.activation_layers:
        return None

    # Pick the layer to use
    if layer is not None:
        layer_key = str(layer)
    else:
        layer_key = next(iter(record.activation_layers))

    if layer_key not in record.activation_layers:
        return None

    layer_info = record.activation_layers[layer_key]
    act_path = Path(layer_info["file"])
    key = str(act_path)
    if key not in readers:
        if not act_path.exists():
            return None
        readers[key] = ActivationReader(act_path)

    reader = readers[key]
    raw = reader.read(layer_info["offset"], layer_info["length"])
    raw_t = torch.from_numpy(raw.astype(np.float32)).to(device)

    with torch.no_grad():
        _, topk_latents, _ = sae(raw_t)
        # Mean-pool across tokens → (d_sae,)
        mean_features = topk_latents.mean(dim=0).cpu().numpy()

    return mean_features


def train_probe(
    sae_checkpoint: Path,
    generations_dir: Path,
    activations_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    layer: int | None = None,
) -> Path:
    """Train a logistic regression probe on SAE features to predict pass/fail.

    Saves:
      - probe_direction.pt: steering direction in hidden-state space (d_model,)
        in the same format as steering_directions.pt for run_experiment.py
      - probe_stats.json: probe accuracy, AUC, top feature weights
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SAE from %s", sae_checkpoint)
    sae = _load_sae(sae_checkpoint, device)
    d_sae = sae.W_dec.shape[0]
    d_model = sae.W_dec.shape[1]

    logger.info("Loading generation records from %s", generations_dir)
    records = _load_all_records(generations_dir)
    logger.info("Loaded %d records", len(records))

    # Build feature matrix X (n_records, d_sae) and labels y (n_records,)
    readers: dict[str, ActivationReader] = {}
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    skipped = 0

    for i, record in enumerate(records):
        if i % 2000 == 0:
            logger.info("Encoding record %d / %d", i, len(records))

        features = _get_record_sae_features(record, sae, readers, device, layer=layer)
        if features is None:
            skipped += 1
            continue

        X_list.append(features)
        y_list.append(1 if record.passed else 0)

    logger.info("Encoded %d records (%d skipped)", len(X_list), skipped)

    X = np.stack(X_list, axis=0)  # (n, d_sae)
    y = np.array(y_list)          # (n,)

    n_pass = y.sum()
    n_fail = len(y) - n_pass
    logger.info("Pass: %d, Fail: %d (%.1f%% pass rate)", n_pass, n_fail, 100 * n_pass / len(y))

    # Train logistic regression on GPU (much faster than sklearn for 16k features)
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)

    # Simple linear probe: w @ x + b → sigmoid → BCE loss + L2 reg
    probe_w = torch.zeros(d_sae, device=device, requires_grad=True)
    probe_b = torch.zeros(1, device=device, requires_grad=True)
    weight_decay = 1e-3
    lr_probe = 1e-2
    n_epochs = 200
    batch_sz = 4096

    optimizer = torch.optim.Adam([probe_w, probe_b], lr=lr_probe)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    logger.info("Training logistic probe on GPU (%d epochs, wd=%.4f)...", n_epochs, weight_decay)

    n = X_t.shape[0]
    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_sz):
            idx = perm[i : i + batch_sz]
            logits = X_t[idx] @ probe_w + probe_b
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y_t[idx])
            loss = loss + weight_decay * probe_w.pow(2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                logits_all = X_t @ probe_w + probe_b
                preds = (logits_all > 0).float()
                acc = (preds == y_t).float().mean().item()
                probs = torch.sigmoid(logits_all)
            logger.info("  epoch %d: loss=%.4f, acc=%.4f", epoch, epoch_loss / n_batches, acc)

    # Final evaluation
    with torch.no_grad():
        logits_all = X_t @ probe_w + probe_b
        preds = (logits_all > 0).float()
        train_acc = (preds == y_t).float().mean().item()
        probs = torch.sigmoid(logits_all)
        # AUC
        y_np = y_t.cpu().numpy()
        p_np = probs.cpu().numpy()
        from sklearn.metrics import roc_auc_score
        train_auc = roc_auc_score(y_np, p_np)

    logger.info("Probe train accuracy: %.3f", train_acc)
    logger.info("Probe train AUC: %.3f", train_auc)

    # Extract weight vector in SAE feature space: w ∈ R^{d_sae}
    w_sae = probe_w.detach().cpu().numpy()

    # Project back to hidden-state space: direction = w @ W_dec → (d_model,)
    # W_dec is (d_sae, d_model), so w_sae @ W_dec gives (d_model,)
    W_dec_np = sae.W_dec.detach().cpu().numpy()
    direction_np = w_sae @ W_dec_np  # (d_model,)
    direction = torch.from_numpy(direction_np.astype(np.float32))

    # Also compute per-variant probe directions (train only on that variant's records)
    # This gives variant-specific steering directions
    variant_directions = {}
    variant_ids = sorted(set(r.variant_id for r in records if r.variant_id))
    for variant in variant_ids:
        variant_mask = np.array([
            i for i, r in enumerate(records)
            if r.variant_id == variant and i < len(X_list)
        ])
        if len(variant_mask) < 50:
            continue
        # Map record indices to X indices (accounting for skipped records)
        # Since we built X_list in order, matching indices work if no skips
        # For safety, rebuild with a variant filter
    # Skip per-variant for now — the global probe direction is the main output

    # Top features by absolute weight
    top_k = 20
    top_indices = np.argsort(np.abs(w_sae))[::-1][:top_k]
    top_features = [
        {"feature_idx": int(idx), "weight": float(w_sae[idx])}
        for idx in top_indices
    ]

    # Save direction in run_experiment.py compatible format
    # Using contrastive format: {"probe_pass_fail": tensor}
    direction_path = output_dir / "probe_direction.pt"
    torch.save({
        "probe_pass_fail": direction,
    }, direction_path)
    logger.info("Saved probe direction to %s (norm=%.4f)", direction_path, direction.norm().item())

    # Save stats
    stats = {
        "train_accuracy": round(train_acc, 4),
        "train_auc": round(train_auc, 4),
        "weight_decay": weight_decay,
        "n_epochs": n_epochs,
        "n_records": len(X_list),
        "n_pass": int(n_pass),
        "n_fail": int(n_fail),
        "d_sae": d_sae,
        "d_model": d_model,
        "direction_norm_before_normalize": round(float(direction.norm()), 4),
        "top_features": top_features,
    }
    stats_path = output_dir / "probe_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved probe stats to %s", stats_path)

    return direction_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train pass/fail probe on SAE features")
    parser.add_argument("--sae-checkpoint", type=Path, required=True)
    parser.add_argument("--generations-dir", type=Path, default=config.GENERATIONS_DIR)
    parser.add_argument("--activations-dir", type=Path, default=config.ACTIVATIONS_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.ANALYSIS_DIR)
    parser.add_argument("--layer", type=int, default=None, help="Layer to use for activations")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    direction_path = train_probe(
        sae_checkpoint=args.sae_checkpoint,
        generations_dir=args.generations_dir,
        activations_dir=args.activations_dir,
        output_dir=args.output_dir,
        device=args.device,
        layer=args.layer,
    )
    print(f"\nProbe direction saved to: {direction_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
