"""Activation Patching: clamp the probe direction component to test causality.

Instead of additive steering (hidden += alpha * direction), this experiment
REPLACES the projection of hidden_states onto the probe direction with a
calibrated target value. This is a stronger intervention that tests whether
the directional information found by the probe is causally involved in
determining code correctness.

Two-phase experiment:
  Phase 1 (baseline): Generate normally, record per-task projections onto
      the probe direction at each decode step.
  Phase 2 (patching): Re-generate with direction component clamped to:
      - clamp_to_pass: mean projection from passing baseline tasks
      - clamp_to_fail: mean projection from failing baseline tasks
      - zero_direction: remove all signal along the direction (projection=0)

If clamping to pass_mean makes failing tasks pass → direction is causally sufficient.
If clamping to fail_mean makes passing tasks fail → direction is causally necessary.
If neither changes outcomes → direction is a readout, not a control signal.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m experiments.steering.activation_patching \
      --direction /scratch/ministral-8b/analysis/layer_27/probe_direction.pt \
      --steer-layer 27 --shard 0 --num-shards 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments import config
from experiments.datasets.load_humaneval import load_humaneval
from experiments.evaluation.executor import execute_code
from experiments.evaluation.extractor import extract_code
from experiments.evaluation.judge import build_humaneval_test_script, classify_failure
from experiments.prompts.builder import build_humaneval_prompt
from experiments.storage.schema import make_generation_record, write_records


# ── Hook factories ────────────────────────────────────────────────────

def make_recording_hook(direction: torch.Tensor, projections: list[float]):
    """Record projections onto the direction at each decode step (no modification).

    Args:
        direction: Unit direction vector, shape (hidden_dim,), on device.
        projections: List to append projection scalars to (mutated in-place).
    """

    def hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output

        if hidden_states.shape[1] == 1:  # decode step only
            proj = torch.sum(
                hidden_states.squeeze() * direction
            ).item()
            projections.append(proj)

        return output

    return hook


def make_clamp_hook(direction: torch.Tensor, target: float):
    """Clamp the projection onto direction to a fixed target value.

    At each decode step, computes the current projection p = h·d,
    then sets it to target by adding (target - p) * d to hidden_states.
    This replaces the direction component rather than just shifting it.

    Args:
        direction: Unit direction vector, shape (hidden_dim,), on device.
        target: Target projection value (scalar).
    """

    def hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output

        if hidden_states.shape[1] == 1:  # decode step only
            # Current projection: scalar per batch element
            proj = torch.sum(hidden_states * direction, dim=-1, keepdim=True)  # (B, 1, 1)
            delta = target - proj  # (B, 1, 1)
            hidden_states = hidden_states + delta * direction.to(hidden_states.dtype)
            if is_tuple:
                return (hidden_states,) + output[1:]
            return hidden_states

        return output

    return hook


# ── Generation ────────────────────────────────────────────────────────

def _generate_text(model, tokenizer, prompt, device):
    """Generate text from a chat-templated prompt."""
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    prompt_tokens = input_ids.shape[1]
    torch.manual_seed(config.BASE_SEED)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = output_ids[0, prompt_tokens:].tolist()
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return generated_text, gen_ids, prompt_tokens


# ── Evaluation ────────────────────────────────────────────────────────

def _evaluate(extracted_code, task):
    """Build test script, execute, classify."""
    test_script = build_humaneval_test_script(
        extracted_code, task["test"], task["entry_point"]
    )
    passed, stderr, exit_code = execute_code(
        test_script, timeout=config.EXTRACTION_TIMEOUT
    )
    return (passed, *classify_failure(passed, stderr, exit_code, extracted_code))


# ── Load direction ────────────────────────────────────────────────────

def _load_direction(path: Path) -> torch.Tensor:
    """Load probe direction from .pt file. Returns unit vector."""
    data = torch.load(path, map_location="cpu", weights_only=True)

    # Handle both formats: {"probe_pass_fail": tensor} or {"directions": {...}}
    if "probe_pass_fail" in data:
        d = data["probe_pass_fail"]
    elif "directions" in data:
        # Take first direction
        key = next(iter(data["directions"]))
        d = data["directions"][key]
    else:
        # Contrastive format: take first non-meta key
        for k, v in data.items():
            if k not in ("norms", "random_directions"):
                d = v
                break
        else:
            raise ValueError(f"No direction found in {path}")

    d = d.float()
    d = d / d.norm()
    print(f"Loaded direction from {path}: shape={d.shape}, norm={d.norm().item():.4f}")
    return d


# ── Main experiment ───────────────────────────────────────────────────

def run_patching_experiment(
    direction_path: Path,
    experiment_name: str,
    output_dir: Path,
    model_path: str,
    steer_layer: int,
    shard: int,
    num_shards: int,
    device: str,
) -> Path:
    t_start = time.time()

    # --- Load direction ---
    direction_cpu = _load_direction(direction_path)

    # --- Load model + tokenizer ---
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    direction_device = direction_cpu.to(device)

    # --- Load and shard tasks ---
    all_tasks = load_humaneval()
    my_tasks = all_tasks[shard::num_shards]
    total_tasks = len(my_tasks)
    print(f"Shard {shard}/{num_shards}: {total_tasks}/{len(all_tasks)} tasks")

    # --- Prepare output ---
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{experiment_name}_shard_{shard}.jsonl"
    if output_path.exists():
        output_path.unlink()

    # ==================================================================
    # PHASE 1: Baseline with projection recording
    # ==================================================================
    print(f"\n{'='*60}")
    print("Phase 1: Baseline (recording projections)")
    print(f"{'='*60}")

    records: list = []
    task_projections: dict[str, list[float]] = {}  # task_id -> [proj per decode step]

    for i, task in enumerate(my_tasks):
        print(f"  Task {i+1}/{total_tasks}: {task['task_id']}")

        # Set up recording hook
        proj_buffer: list[float] = []
        handle = model.model.layers[steer_layer].register_forward_hook(
            make_recording_hook(direction_device, proj_buffer)
        )

        prompt = build_humaneval_prompt(task, "baseline")
        generated_text, gen_ids, prompt_tokens = _generate_text(
            model, tokenizer, prompt, device
        )
        handle.remove()

        # Store projections for this task
        task_projections[task["task_id"]] = proj_buffer

        extracted_code, extraction_clean = extract_code(
            generated_text, task["entry_point"]
        )
        passed, failure_category, error_message, error_hash = _evaluate(
            extracted_code, task
        )

        record = make_generation_record(
            task_id=task["task_id"],
            dataset=task["dataset"],
            variant_id="baseline",
            run_id=0,
            seed=config.BASE_SEED,
            prompt_text=prompt,
            prompt_tokens=prompt_tokens,
            generated_text=generated_text,
            extracted_code=extracted_code,
            gen_token_ids=gen_ids,
            generated_tokens=len(gen_ids),
            extraction_clean=extraction_clean,
            passed=passed,
            failure_category=failure_category,
            error_message=error_message,
            error_hash=error_hash,
        )
        records.append(record)

    # --- Compute projection statistics ---
    baseline_pass_ids = {r.task_id for r in records if r.passed}
    baseline_fail_ids = {r.task_id for r in records if not r.passed}

    pass_proj_means = [
        np.mean(task_projections[tid])
        for tid in baseline_pass_ids
        if task_projections[tid]
    ]
    fail_proj_means = [
        np.mean(task_projections[tid])
        for tid in baseline_fail_ids
        if task_projections[tid]
    ]

    mu_pass = float(np.mean(pass_proj_means)) if pass_proj_means else 0.0
    mu_fail = float(np.mean(fail_proj_means)) if fail_proj_means else 0.0
    all_proj_means = pass_proj_means + fail_proj_means
    sigma = float(np.std(all_proj_means)) if all_proj_means else 1.0

    baseline_pass = len(baseline_pass_ids)
    baseline_rate = baseline_pass / max(total_tasks, 1)

    print(f"\n  Baseline: {baseline_pass}/{total_tasks} ({100*baseline_rate:.1f}%)")
    print(f"  Projection stats (mean per generation):")
    print(f"    Pass mean: {mu_pass:.4f} (n={len(pass_proj_means)})")
    print(f"    Fail mean: {mu_fail:.4f} (n={len(fail_proj_means)})")
    print(f"    Separation: {mu_pass - mu_fail:.4f}")
    print(f"    Pooled std: {sigma:.4f}")
    if sigma > 0:
        print(f"    Cohen's d: {(mu_pass - mu_fail) / sigma:.4f}")

    # ==================================================================
    # PHASE 2: Clamped conditions
    # ==================================================================
    clamp_conditions = [
        ("clamp_to_pass", mu_pass),
        ("clamp_to_fail", mu_fail),
        ("zero_direction", 0.0),
    ]

    for cond_name, target_value in clamp_conditions:
        print(f"\n{'='*60}")
        print(f"Phase 2: {cond_name} (target={target_value:.4f})")
        print(f"{'='*60}")

        handle = model.model.layers[steer_layer].register_forward_hook(
            make_clamp_hook(direction_device, target_value)
        )

        for i, task in enumerate(my_tasks):
            print(f"  Task {i+1}/{total_tasks}: {task['task_id']}")

            prompt = build_humaneval_prompt(task, "baseline")
            generated_text, gen_ids, prompt_tokens = _generate_text(
                model, tokenizer, prompt, device
            )

            extracted_code, extraction_clean = extract_code(
                generated_text, task["entry_point"]
            )
            passed, failure_category, error_message, error_hash = _evaluate(
                extracted_code, task
            )

            record = make_generation_record(
                task_id=task["task_id"],
                dataset=task["dataset"],
                variant_id=cond_name,
                run_id=0,
                seed=config.BASE_SEED,
                prompt_text=prompt,
                prompt_tokens=prompt_tokens,
                generated_text=generated_text,
                extracted_code=extracted_code,
                gen_token_ids=gen_ids,
                generated_tokens=len(gen_ids),
                extraction_clean=extraction_clean,
                passed=passed,
                failure_category=failure_category,
                error_message=error_message,
                error_hash=error_hash,
            )
            records.append(record)

        handle.remove()

        cond_pass = sum(1 for r in records if r.variant_id == cond_name and r.passed)
        cond_rate = cond_pass / max(total_tasks, 1)
        delta = cond_rate - baseline_rate
        print(f"  {cond_name}: {cond_pass}/{total_tasks} ({100*cond_rate:.1f}%), delta={100*delta:+.1f}%")

        # Show per-task transitions
        transitions = {"pass->fail": 0, "fail->pass": 0, "same": 0}
        for task in my_tasks:
            tid = task["task_id"]
            bl = any(r.passed for r in records if r.variant_id == "baseline" and r.task_id == tid)
            cl = any(r.passed for r in records if r.variant_id == cond_name and r.task_id == tid)
            if bl and not cl:
                transitions["pass->fail"] += 1
            elif not bl and cl:
                transitions["fail->pass"] += 1
            else:
                transitions["same"] += 1
        print(f"  Transitions: {transitions}")

    # --- Save ---
    print(f"\nSaving {len(records)} records to {output_path}")
    write_records(output_path, records)

    # --- Final summary ---
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print("Activation Patching Summary")
    print(f"{'='*60}")
    print(f"  Direction: {direction_path}")
    print(f"  Layer: {steer_layer}")
    print(f"  Pass projection mean: {mu_pass:.4f}")
    print(f"  Fail projection mean: {mu_fail:.4f}")
    print(f"  Baseline: {baseline_pass}/{total_tasks} ({100*baseline_rate:.1f}%)")

    for cond_name, target_value in clamp_conditions:
        cond_records = [r for r in records if r.variant_id == cond_name]
        cond_pass = sum(1 for r in cond_records if r.passed)
        cond_rate = cond_pass / max(len(cond_records), 1)
        delta = cond_rate - baseline_rate
        print(f"  {cond_name} (target={target_value:.4f}): {cond_pass}/{len(cond_records)} ({100*cond_rate:.1f}%), delta={100*delta:+.1f}%")

    print(f"  Elapsed: {elapsed:.1f}s")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Activation patching: clamp probe direction to test causality.",
    )
    parser.add_argument(
        "--direction", type=str, required=True,
        help="Path to .pt file with probe direction",
    )
    parser.add_argument(
        "--experiment-name", type=str, default="activation_patching",
        help="Name for the experiment (default: activation_patching)",
    )
    config.add_model_arg(parser)
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: /scratch/{model}/steering/patching)",
    )
    parser.add_argument(
        "--steer-layer", type=int, required=True,
        help="Decoder layer index for the patching hook",
    )
    parser.add_argument(
        "--shard", type=int, required=True,
        help="Shard index (0-based)",
    )
    parser.add_argument(
        "--num-shards", type=int, default=2,
        help="Total shards (default: 2)",
    )
    args = parser.parse_args()

    config.apply_args(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, running on CPU (will be very slow)")

    output_dir = args.output_dir or str(config.STEERING_DIR / "patching")

    output_path = run_patching_experiment(
        direction_path=Path(args.direction),
        experiment_name=args.experiment_name,
        output_dir=Path(output_dir),
        model_path=config.MODEL_ID,
        steer_layer=args.steer_layer,
        shard=args.shard,
        num_shards=args.num_shards,
        device=device,
    )

    print(f"\nOutput: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
