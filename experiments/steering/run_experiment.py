"""Generic steering experiment runner. For each direction x alpha x task, generate code
with a steering hook, extract, evaluate, and save.

Reads: HumanEval dataset, steering directions (.pt file), HF model
Writes: /scratch/steering/{experiment_name}_shard_{N}.jsonl

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m experiments.steering.run_experiment \
      --directions /scratch/sae/directions.pt --experiment-name sae_steering --shard 0
  CUDA_VISIBLE_DEVICES=1 python -m experiments.steering.run_experiment \
      --directions /scratch/sae/directions.pt --experiment-name sae_steering --shard 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments import config

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


def _wandb_enabled() -> bool:
    if _wandb is None:
        return False
    return os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1")
from experiments.datasets.load_humaneval import load_humaneval
from experiments.evaluation.executor import execute_code
from experiments.evaluation.extractor import extract_code
from experiments.evaluation.judge import build_humaneval_test_script, classify_failure
from experiments.prompts.builder import build_humaneval_prompt
from experiments.steering.hook import make_steering_hook
from experiments.storage.schema import make_generation_record, write_records


def _load_directions(
    directions_path: Path,
    include_random_controls: bool,
) -> list[tuple[str, torch.Tensor]]:
    """Load steering directions from a .pt file.

    Supports two formats:
    - SAE format: {"directions": {int_key: tensor}, "random_directions": {int_key: tensor}}
    - Contrastive format: {"typed": tensor, "invariants": tensor, ...} (skip "norms" key)

    Returns:
        List of (name, direction) pairs, each direction normalized to unit length.
    """
    data = torch.load(directions_path, map_location="cpu", weights_only=True)

    pairs: list[tuple[str, torch.Tensor]] = []

    if "directions" in data:
        # SAE format
        for key, tensor in data["directions"].items():
            direction = tensor / tensor.norm()
            pairs.append((str(key), direction))

        if include_random_controls and "random_directions" in data:
            for key, tensor in data["random_directions"].items():
                direction = tensor / tensor.norm()
                pairs.append((f"random_{key}", direction))
    else:
        # Contrastive format
        for key, tensor in data.items():
            if key in ("norms", "random_directions"):
                continue
            direction = tensor / tensor.norm()
            pairs.append((str(key), direction))

        if include_random_controls and "random_directions" in data:
            for key, tensor in data["random_directions"].items():
                direction = tensor / tensor.norm()
                pairs.append((f"random_{key}", direction))

    print(f"Loaded {len(pairs)} steering directions from {directions_path}")
    for name, d in pairs:
        print(f"  {name}: shape={d.shape}, norm={d.norm().item():.4f}")

    return pairs


def _generate_text(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: str,
) -> tuple[str, list[int], int]:
    """Generate text from a chat-templated prompt.

    Returns:
        (generated_text, gen_token_ids, prompt_token_count)
    """
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


def _evaluate_task(
    extracted_code: str,
    task: dict,
) -> tuple[bool, str, str, str]:
    """Build test script, execute, and classify the result.

    Returns:
        (passed, failure_category, error_message, error_hash)
    """
    test_script = build_humaneval_test_script(
        extracted_code, task["test"], task["entry_point"]
    )
    passed, stderr, exit_code = execute_code(test_script, timeout=config.EXTRACTION_TIMEOUT)
    failure_category, error_message, error_hash = classify_failure(
        passed, stderr, exit_code, extracted_code
    )
    return passed, failure_category, error_message, error_hash


def run_experiment(
    directions_path: Path,
    experiment_name: str,
    output_dir: Path,
    model_path: str,
    alphas: list[float],
    steer_layer: int,
    shard: int,
    num_shards: int,
    include_random_controls: bool,
    device: str,
) -> Path:
    """Run a steering experiment across directions, alphas, and HumanEval tasks.

    Args:
        directions_path: Path to .pt file with steering directions.
        experiment_name: Name for the output file prefix.
        output_dir: Directory to write JSONL output.
        model_path: HuggingFace model ID or local path.
        alphas: List of steering magnitudes.
        steer_layer: Decoder layer index for hook attachment.
        shard: This worker's shard index (0-based).
        num_shards: Total number of shards.
        include_random_controls: Whether to include random direction controls.
        device: Torch device string (e.g., "cuda").

    Returns:
        Path to the output JSONL file.
    """
    t_start = time.time()

    # --- wandb ---
    use_wandb = _wandb_enabled()
    if use_wandb:
        _wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name=f"{experiment_name}_shard_{shard}",
            config={
                "experiment_name": experiment_name,
                "alphas": alphas, "steer_layer": steer_layer,
                "shard": shard, "num_shards": num_shards,
                "include_random_controls": include_random_controls,
            },
        )
        print("  wandb: logging enabled")

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

    # --- Load steering directions ---
    directions = _load_directions(directions_path, include_random_controls)

    # --- Load and shard tasks ---
    all_tasks = load_humaneval()
    my_tasks = all_tasks[shard::num_shards]
    print(f"Shard {shard}/{num_shards}: {len(my_tasks)}/{len(all_tasks)} tasks")

    # --- Prepare output ---
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{experiment_name}_shard_{shard}.jsonl"
    if output_path.exists():
        output_path.unlink()

    records: list = []
    total_tasks = len(my_tasks)
    total_conditions = 1 + len(directions) * len(alphas)  # baseline + steered
    print(f"Conditions per task: 1 baseline + {len(directions)} dirs x {len(alphas)} alphas = {total_conditions}")
    print(f"Total generations: {total_conditions * total_tasks}")

    # === Baseline: no steering ===
    print(f"\n{'='*60}")
    print("Running baseline (no steering)")
    print(f"{'='*60}")

    for i, task in enumerate(my_tasks):
        print(f"  Task {i+1}/{total_tasks}: {task['task_id']}, condition baseline")

        prompt = build_humaneval_prompt(task, "baseline")
        generated_text, gen_ids, prompt_tokens = _generate_text(
            model, tokenizer, prompt, device
        )

        extracted_code, extraction_clean = extract_code(
            generated_text, task["entry_point"]
        )

        passed, failure_category, error_message, error_hash = _evaluate_task(
            extracted_code, task
        )

        record = make_generation_record(
            task_id=task["task_id"],
            dataset=task["dataset"],
            variant_id="baseline_no_steer",
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

    baseline_pass = sum(1 for r in records if r.passed)
    print(f"Baseline: {baseline_pass}/{total_tasks} passed")

    if use_wandb:
        _wandb.log({
            "baseline_pass_rate": baseline_pass / max(total_tasks, 1),
            "baseline_passed": baseline_pass,
            "baseline_n": total_tasks,
        })

    # === Steered conditions: direction x alpha ===
    for dir_idx, (dir_name, direction) in enumerate(directions):
        for alpha in alphas:
            condition = f"steer_{dir_name}_alpha_{alpha}"
            print(f"\n{'='*60}")
            print(f"Direction {dir_idx+1}/{len(directions)}: {dir_name}, alpha={alpha}")
            print(f"{'='*60}")

            direction_device = direction.to(device)

            for i, task in enumerate(my_tasks):
                print(f"  Task {i+1}/{total_tasks}: {task['task_id']}, condition {condition}")

                prompt = build_humaneval_prompt(task, "baseline")

                # Attach steering hook
                handle = model.model.layers[steer_layer].register_forward_hook(
                    make_steering_hook(direction_device, alpha)
                )

                generated_text, gen_ids, prompt_tokens = _generate_text(
                    model, tokenizer, prompt, device
                )

                # Remove hook immediately after generation
                handle.remove()

                extracted_code, extraction_clean = extract_code(
                    generated_text, task["entry_point"]
                )

                passed, failure_category, error_message, error_hash = _evaluate_task(
                    extracted_code, task
                )

                record = make_generation_record(
                    task_id=task["task_id"],
                    dataset=task["dataset"],
                    variant_id=condition,
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

            steered_pass = sum(
                1 for r in records
                if r.variant_id == condition and r.passed
            )
            steered_rate = steered_pass / max(total_tasks, 1)
            baseline_rate = baseline_pass / max(total_tasks, 1)
            print(f"  {condition}: {steered_pass}/{total_tasks} passed")

            if use_wandb:
                _wandb.log({
                    f"pass_rate/{condition}": steered_rate,
                    f"delta/{condition}": steered_rate - baseline_rate,
                })

    # === Save all records ===
    print(f"\nSaving {len(records)} records to {output_path}")
    write_records(output_path, records)

    # === Summary ===
    elapsed = time.time() - t_start
    pass_count = sum(1 for r in records if r.passed)
    categories = {}
    for r in records:
        categories[r.failure_category] = categories.get(r.failure_category, 0) + 1

    print(f"\n{'='*60}")
    print(f"Experiment complete: {experiment_name} shard {shard}")
    print(f"{'='*60}")
    print(f"  Total records: {len(records)}")
    print(f"  Overall pass rate: {pass_count}/{len(records)} ({100*pass_count/len(records):.1f}%)")
    print(f"  Failure breakdown: {categories}")
    print(f"  Elapsed: {elapsed:.1f}s")

    if use_wandb:
        _wandb.log({
            "total_records": len(records),
            "overall_pass_rate": pass_count / max(len(records), 1),
            "elapsed_s": elapsed,
        })
        _wandb.finish()

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a steering experiment: generate code with steering hooks, evaluate, save.",
    )
    parser.add_argument(
        "--directions", type=str, required=True,
        help="Path to .pt file with steering directions",
    )
    parser.add_argument(
        "--experiment-name", type=str, required=True,
        help="Name for the experiment (e.g., 'sae_steering')",
    )
    parser.add_argument(
        "--model-path", type=str, default=config.MODEL_ID,
        help=f"HuggingFace model ID or local path (default: {config.MODEL_ID})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(config.STEERING_DIR),
        help=f"Output directory for JSONL files (default: {config.STEERING_DIR})",
    )
    parser.add_argument(
        "--alphas", type=str, default="3.0,-3.0",
        help="Comma-separated alpha values (default: '3.0,-3.0')",
    )
    parser.add_argument(
        "--steer-layer", type=int, default=config.CAPTURE_LAYER,
        help=f"Decoder layer index for steering hook (default: {config.CAPTURE_LAYER})",
    )
    parser.add_argument(
        "--shard", type=int, required=True,
        help="Shard index for this worker (0-based)",
    )
    parser.add_argument(
        "--num-shards", type=int, default=2,
        help="Total number of shards (default: 2)",
    )
    parser.add_argument(
        "--include-random-controls", action="store_true",
        help="Include random direction controls from the directions file",
    )
    args = parser.parse_args()

    alphas = [float(a.strip()) for a in args.alphas.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, running on CPU (will be very slow)")

    output_path = run_experiment(
        directions_path=Path(args.directions),
        experiment_name=args.experiment_name,
        output_dir=Path(args.output_dir),
        model_path=args.model_path,
        alphas=alphas,
        steer_layer=args.steer_layer,
        shard=args.shard,
        num_shards=args.num_shards,
        include_random_controls=args.include_random_controls,
        device=device,
    )

    print(f"\nOutput: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
