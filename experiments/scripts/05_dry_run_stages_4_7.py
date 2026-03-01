"""Dry run for stages 4-7: exercises SAE training, analysis, contrastive directions,
steering experiment, and results analysis with synthetic data. No GPU required.

Creates fake generation records + activations in a temp directory, then runs each
stage's core function with tiny parameters (d_model=64, d_sae=128, K=4).

What this catches:
  - Import errors in all new modules
  - Schema field mismatches (e.g., analyze reading fields that run_experiment writes)
  - torch.save/load format compatibility between stages
  - Direction normalization and steering hook integration
  - JSONL round-trip through steering records
  - analyze_steering parsing of variant_id patterns

Usage:
  python3 -m experiments.scripts.05_dry_run_stages_4_7
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

# --- Tiny dimensions so it runs on CPU in seconds ---
D_MODEL = 64
D_SAE = 128
K = 4
NUM_TASKS = 4
NUM_VARIANTS = 3  # baseline + 2 variants
BATCH_SIZE = 8
TOKEN_BUDGET = 64  # tiny
TOKENS_PER_RECORD = 8

# --- Patch config BEFORE any pipeline module imports ---
# This ensures ActivationReader/Writer use D_MODEL=64 instead of 4096
import experiments.config as _cfg
_cfg.MODEL_HIDDEN_DIM = D_MODEL
_cfg.VARIANT_IDS = ["baseline", "typed", "invariants"]

# Force-patch module-level constants that cache MODEL_HIDDEN_DIM at import time
import experiments.storage.activation_store as _act_store
_act_store.HIDDEN_DIM = D_MODEL
_act_store.BYTES_PER_ROW = D_MODEL * 2

import experiments.contrastive.compute_directions as _contrastive
_contrastive.HIDDEN_DIM = D_MODEL


def _print_result(label: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")


def _create_fake_data(tmp_dir: Path) -> tuple[Path, Path]:
    """Create fake generation records + activation shards mimicking stages 1-3 output.

    Returns (generations_dir, activations_dir).
    """
    from experiments.storage.schema import make_generation_record, write_records

    gen_dir = tmp_dir / "generations"
    gen_dir.mkdir()
    act_dir = tmp_dir / "activations"
    act_dir.mkdir()

    variants = ["baseline", "typed", "invariants"]
    records = []
    act_offset = 0

    act_path = act_dir / "shard_0.npy"

    # Build all activations in memory, then save
    all_activations = []

    for task_idx in range(NUM_TASKS):
        for variant_id in variants:
            # Alternate pass/fail so stratified loader gets both
            passed = (task_idx + hash(variant_id)) % 2 == 0

            token_ids = list(range(TOKENS_PER_RECORD))
            activations = np.random.randn(TOKENS_PER_RECORD, D_MODEL).astype(np.float16)
            all_activations.append(activations)

            record = make_generation_record(
                task_id=f"HumanEval/{task_idx}",
                dataset="humaneval",
                variant_id=variant_id,
                run_id=0,
                seed=42,
                prompt_text=f"Write function_{task_idx}",
                prompt_tokens=10,
                generated_text=f"def function_{task_idx}(): pass",
                extracted_code=f"def function_{task_idx}(): pass",
                gen_token_ids=token_ids,
                generated_tokens=len(token_ids),
                extraction_clean=True,
                passed=passed,
                failure_category="pass" if passed else "wrong_answer",
                error_message="" if passed else "AssertionError",
                error_hash="" if passed else "abc123",
                activation_file=str(act_path),
                activation_offset=act_offset,
                activation_length=TOKENS_PER_RECORD,
            )
            records.append(record)
            act_offset += TOKENS_PER_RECORD

    # Write activations as contiguous float16 mmap
    stacked = np.concatenate(all_activations, axis=0)
    stacked.tofile(str(act_path))

    # Write generation records
    write_records(gen_dir / "shard_0.jsonl", records)

    return gen_dir, act_dir


def test_stage4_sae_training(gen_dir: Path, act_dir: Path, output_dir: Path) -> Path:
    """Stage 4: Train SAE on fake activations."""
    print("\n=== Stage 4: SAE Training ===")

    from experiments.storage.schema import read_records
    from experiments.sae.train import train_sae

    records = read_records(gen_dir / "shard_0.jsonl")
    _print_result("Records loaded", len(records) > 0, f"{len(records)} records")

    ckpt_path = train_sae(
        records=records,
        activation_dir=act_dir,
        output_dir=output_dir / "sae",
        batch_size=BATCH_SIZE,
        lr=1e-3,
        device="cpu",
        token_budget=TOKEN_BUDGET,
        d_model=D_MODEL,
        d_sae=D_SAE,
        k=K,
    )

    ckpt_exists = ckpt_path.exists()
    _print_result("Checkpoint saved", ckpt_exists, str(ckpt_path))

    # Verify checkpoint loads correctly
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    has_keys = "state_dict" in ckpt and "config" in ckpt
    _print_result("Checkpoint has expected keys", has_keys, str(list(ckpt.keys())))

    cfg = ckpt["config"]
    cfg_ok = cfg["d_model"] == D_MODEL and cfg["d_sae"] == D_SAE and cfg["k"] == K
    _print_result("Config matches", cfg_ok, f"d_model={cfg['d_model']}, d_sae={cfg['d_sae']}, k={cfg['k']}")

    return ckpt_path


def test_stage5a_analyze(
    ckpt_path: Path, gen_dir: Path, act_dir: Path, output_dir: Path
) -> Path:
    """Stage 5a: Analyze SAE features."""
    print("\n=== Stage 5a: Analyze Features ===")

    from experiments.sae.analyze import analyze_features

    stats_path = analyze_features(
        sae_checkpoint=ckpt_path,
        generations_dir=gen_dir,
        activations_dir=act_dir,
        output_dir=output_dir / "analysis",
        batch_size=BATCH_SIZE,
        device="cpu",
    )

    stats_exists = stats_path.exists()
    _print_result("Feature stats written", stats_exists, str(stats_path))

    with open(stats_path) as f:
        stats = json.load(f)

    has_features = "features" in stats and len(stats["features"]) > 0
    _print_result("Stats has features", has_features, f"{len(stats.get('features', []))} features")

    # Spot check a feature entry
    feat = stats["features"][0]
    expected_keys = {"feature_idx", "cohens_d", "mean_pass", "mean_fail"}
    has_keys = expected_keys.issubset(feat.keys())
    _print_result("Feature entry has expected keys", has_keys, str(list(feat.keys())[:6]))

    return stats_path


def test_stage5b_select_candidates(
    stats_path: Path, ckpt_path: Path, output_dir: Path
) -> tuple[Path, Path]:
    """Stage 5b: Select steering candidates."""
    print("\n=== Stage 5b: Select Candidates ===")

    from experiments.sae.select_candidates import select_steering_candidates

    candidates_path = select_steering_candidates(
        feature_stats_path=stats_path,
        sae_checkpoint=ckpt_path,
        output_dir=output_dir / "analysis",
        device="cpu",
    )

    cand_exists = candidates_path.exists()
    _print_result("Candidates JSON written", cand_exists, str(candidates_path))

    with open(candidates_path) as f:
        cand_data = json.load(f)

    has_candidates = "candidates" in cand_data
    _print_result("Has candidates key", has_candidates)

    # Check steering directions .pt file
    directions_path = output_dir / "analysis" / "steering_directions.pt"
    dir_exists = directions_path.exists()
    _print_result("Steering directions .pt exists", dir_exists)

    if dir_exists:
        data = torch.load(directions_path, map_location="cpu", weights_only=True)
        has_directions = "directions" in data
        _print_result("Directions .pt has expected format", has_directions, str(list(data.keys())))

    return candidates_path, directions_path


def test_stage5c_contrastive(gen_dir: Path, act_dir: Path, output_dir: Path) -> Path:
    """Stage 5c: Compute contrastive directions."""
    print("\n=== Stage 5c: Contrastive Directions ===")

    from experiments.contrastive.compute_directions import compute_contrastive_directions

    contrastive_path = compute_contrastive_directions(
        generations_dir=gen_dir,
        activations_dir=act_dir,
        output_dir=output_dir / "analysis",
        device="cpu",
    )

    exists = contrastive_path.exists()
    _print_result("Contrastive directions saved", exists, str(contrastive_path))

    data = torch.load(contrastive_path, map_location="cpu", weights_only=True)
    has_norms = "norms" in data
    non_meta_keys = [k for k in data.keys() if k not in ("norms", "random_directions")]
    _print_result("Has variant directions", len(non_meta_keys) > 0, f"variants: {non_meta_keys}")

    # Verify directions are unit norm
    for key in non_meta_keys:
        norm = data[key].norm().item()
        _print_result(f"Direction '{key}' is unit norm", abs(norm - 1.0) < 0.01, f"norm={norm:.4f}")

    return contrastive_path


def test_stage6_steering_loader(sae_directions_path: Path, contrastive_path: Path):
    """Stage 6-7: Test the direction loader from run_experiment.py (no model needed)."""
    print("\n=== Stage 6-7: Steering Direction Loader ===")

    # Import the private loader function
    from experiments.steering.run_experiment import _load_directions

    # Test SAE format loading
    sae_pairs = _load_directions(sae_directions_path, include_random_controls=True)
    _print_result(
        "SAE directions loaded",
        len(sae_pairs) > 0,
        f"{len(sae_pairs)} directions",
    )

    for name, d in sae_pairs:
        norm_ok = abs(d.norm().item() - 1.0) < 0.01
        _print_result(f"  SAE dir '{name}' unit norm", norm_ok, f"shape={d.shape}")

    # Test contrastive format loading
    contrastive_pairs = _load_directions(contrastive_path, include_random_controls=False)
    _print_result(
        "Contrastive directions loaded",
        len(contrastive_pairs) > 0,
        f"{len(contrastive_pairs)} directions",
    )

    for name, d in contrastive_pairs:
        norm_ok = abs(d.norm().item() - 1.0) < 0.01
        _print_result(f"  Contrastive dir '{name}' unit norm", norm_ok, f"shape={d.shape}")


def test_steering_hook_integration():
    """Test that a steering direction can be applied via the hook."""
    print("\n=== Steering Hook Integration ===")

    from experiments.steering.hook import make_steering_hook

    direction = torch.randn(D_MODEL)
    direction = direction / direction.norm()
    alpha = 3.0

    hook_fn = make_steering_hook(direction, alpha)
    _print_result("Hook created", hook_fn is not None)

    # Simulate a decoder layer output: (batch=1, seq_len=1, d_model)
    # shape[1]==1 means decode step, hook should apply
    fake_output = torch.zeros(1, 1, D_MODEL)
    modified = hook_fn(None, None, (fake_output,))

    if modified is not None:
        diff = (modified[0] - fake_output).norm().item()
        expected_diff = alpha  # direction is unit norm
        close = abs(diff - expected_diff) < 0.1
        _print_result("Hook applied steering", close, f"diff_norm={diff:.4f}, expected={expected_diff}")
    else:
        # Hook modifies in place
        diff = fake_output.norm().item()
        close = abs(diff - alpha) < 0.1
        _print_result("Hook applied steering (in-place)", close, f"output_norm={diff:.4f}")


def test_stage8_analyze_steering(tmp_dir: Path, output_dir: Path):
    """Stage 8: Test analyze_steering with synthetic steering records."""
    print("\n=== Stage 8: Analyze Steering ===")

    from experiments.storage.schema import make_generation_record, write_records
    from experiments.steering.analyze_steering import analyze_steering

    steering_dir = tmp_dir / "steering"
    steering_dir.mkdir(exist_ok=True)

    # Create fake SAE steering records
    records = []
    for task_idx in range(4):
        # Baseline
        records.append(make_generation_record(
            task_id=f"HumanEval/{task_idx}",
            dataset="humaneval",
            variant_id="baseline_no_steer",
            run_id=0, seed=42,
            prompt_text="...", prompt_tokens=10,
            generated_text="...", extracted_code="...",
            passed=task_idx < 2,  # 50% baseline
            failure_category="pass" if task_idx < 2 else "wrong_answer",
        ))
        # Steered condition
        records.append(make_generation_record(
            task_id=f"HumanEval/{task_idx}",
            dataset="humaneval",
            variant_id="steer_42_alpha_3.0",
            run_id=0, seed=42,
            prompt_text="...", prompt_tokens=10,
            generated_text="...", extracted_code="...",
            passed=task_idx < 3,  # 75% steered
            failure_category="pass" if task_idx < 3 else "wrong_answer",
        ))

    write_records(steering_dir / "sae_steering_shard_0.jsonl", records)

    # Also test contrastive format
    contrastive_records = []
    for task_idx in range(4):
        contrastive_records.append(make_generation_record(
            task_id=f"HumanEval/{task_idx}",
            dataset="humaneval",
            variant_id="baseline_no_steer",
            run_id=0, seed=42,
            prompt_text="...", prompt_tokens=10,
            passed=task_idx < 2,
            failure_category="pass" if task_idx < 2 else "wrong_answer",
        ))
        contrastive_records.append(make_generation_record(
            task_id=f"HumanEval/{task_idx}",
            dataset="humaneval",
            variant_id="steer_typed_alpha_3.0",
            run_id=0, seed=42,
            prompt_text="...", prompt_tokens=10,
            passed=task_idx < 1,  # 25% — worse
            failure_category="pass" if task_idx < 1 else "wrong_answer",
        ))

    write_records(steering_dir / "contrastive_steering_shard_0.jsonl", contrastive_records)

    analysis_dir = output_dir / "analysis"
    result_path = analyze_steering(
        steering_dir=steering_dir,
        output_dir=analysis_dir,
    )

    exists = result_path.exists()
    _print_result("steering_results.json written", exists)

    with open(result_path) as f:
        results = json.load(f)

    has_sae = "sae_steering" in results
    has_contrastive = "contrastive_steering" in results
    _print_result("Has SAE results", has_sae)
    _print_result("Has contrastive results", has_contrastive)

    if has_sae:
        sae_r = results["sae_steering"]
        baseline_ok = sae_r["baseline_pass_rate"] == 0.5
        _print_result("SAE baseline rate = 0.5", baseline_ok, f"got {sae_r['baseline_pass_rate']}")

        if sae_r["conditions"]:
            cond = sae_r["conditions"][0]
            delta_ok = cond["delta"] == 0.25  # 0.75 - 0.5
            _print_result("SAE delta = +0.25", delta_ok, f"got {cond['delta']}")

    if has_contrastive:
        con_r = results["contrastive_steering"]
        if con_r["conditions"]:
            cond = con_r["conditions"][0]
            delta_ok = cond["delta"] == -0.25  # 0.25 - 0.5
            _print_result("Contrastive delta = -0.25", delta_ok, f"got {cond['delta']}")


def main() -> int:
    t_start = time.time()
    tmp_dir = Path(tempfile.mkdtemp(prefix="dry_run_4_7_"))
    output_dir = tmp_dir / "output"
    output_dir.mkdir()
    all_ok = True

    print(f"Stages 4-7 dry run — temp dir: {tmp_dir}")
    print(f"Tiny config: d_model={D_MODEL}, d_sae={D_SAE}, K={K}\n")

    try:
        # Create fake stages 1-3 output
        print("=== Creating Fake Data (mimicking stages 1-3) ===")
        gen_dir, act_dir = _create_fake_data(tmp_dir)
        _print_result("Fake data created", True, f"{gen_dir}, {act_dir}")

        # Stage 4
        ckpt_path = test_stage4_sae_training(gen_dir, act_dir, output_dir)

        # Stage 5a
        stats_path = test_stage5a_analyze(ckpt_path, gen_dir, act_dir, output_dir)

        # Stage 5b
        candidates_path, sae_directions_path = test_stage5b_select_candidates(
            stats_path, ckpt_path, output_dir
        )

        # Stage 5c
        contrastive_path = test_stage5c_contrastive(gen_dir, act_dir, output_dir)

        # Stage 6-7: test direction loading (no model required)
        test_stage6_steering_loader(sae_directions_path, contrastive_path)

        # Steering hook integration
        test_steering_hook_integration()

        # Stage 8
        test_stage8_analyze_steering(tmp_dir, output_dir)

    except Exception as e:
        print(f"\n  EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed = time.time() - t_start
    print(f"\nCompleted in {elapsed:.1f}s (temp dir cleaned up)")

    print("\n" + "=" * 60)
    if all_ok:
        print("STAGES 4-7 DRY RUN PASSED")
    else:
        print("STAGES 4-7 DRY RUN FAILED — fix issues before overnight run")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
