"""Stage 2: Evaluate all generated code against HumanEval/MBPP test cases.

Reads: /scratch/<model>/generations/shard_*.jsonl + HumanEval/MBPP datasets
Writes: /scratch/<model>/generations/shard_*.jsonl (updated with evaluation results)

CPU-bound — no GPU needed. Processes all shards sequentially.

Usage:
  python -m experiments.scripts.02_evaluate --model ministral-8b
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import wandb

from experiments import config
from experiments.datasets.load_humaneval import load_humaneval
from experiments.datasets.load_mbpp import load_mbpp, load_mbpp_plus_tests
from experiments.evaluation.executor import execute_code
from experiments.evaluation.judge import (
    build_humaneval_test_script,
    build_mbpp_test_script,
    classify_failure,
)
from experiments.storage.schema import GenerationRecord, read_records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2: Evaluate generated code against test cases.",
    )
    config.add_model_arg(parser)
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Directory containing shard JSONL files (default: model's generations dir)",
    )
    args = parser.parse_args()

    config.apply_args(args)
    input_dir = args.input_dir or config.GENERATIONS_DIR

    # --- W&B ---
    run = wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        name=f"02_evaluate_{config.MODEL_NAME}",
        config={
            "stage": "02_evaluate",
            "model": config.MODEL_NAME,
            "model_id": config.MODEL_ID,
            "extraction_timeout": config.EXTRACTION_TIMEOUT,
        },
    )

    # --- Load test harnesses ---
    print("Loading datasets for test harnesses...")
    humaneval_tasks = {t["task_id"]: t for t in load_humaneval()}
    mbpp_tasks = {t["task_id"]: t for t in load_mbpp()}
    mbpp_plus = load_mbpp_plus_tests()
    print(f"  MBPP+ augmented tests for {len(mbpp_plus)} tasks")

    # --- Find shard files ---
    shard_files = sorted(input_dir.glob("shard_*.jsonl"))
    if not shard_files:
        print(f"No shard files found in {input_dir}")
        return 1
    print(f"Found {len(shard_files)} shard files")

    total_records = 0
    total_passed = 0

    for shard_file in shard_files:
        records = read_records(shard_file)
        print(f"\nEvaluating {shard_file.name}: {len(records)} records")
        t0 = time.time()

        pass_count = 0
        category_counts: dict[str, int] = {}

        for i, record in enumerate(records):
            # Build test script
            if record.dataset == "humaneval":
                task = humaneval_tasks[record.task_id]
                test_script = build_humaneval_test_script(
                    record.extracted_code, task["test"], task["entry_point"]
                )
            else:
                task = mbpp_tasks[record.task_id]
                # Use MBPP+ augmented tests when available (stricter)
                test_list = mbpp_plus.get(record.task_id, task["test_list"])
                test_script = build_mbpp_test_script(
                    record.extracted_code, test_list, task["test_setup_code"]
                )

            # Execute in sandbox
            passed, stderr, exit_code = execute_code(
                test_script, timeout=config.EXTRACTION_TIMEOUT
            )

            # Classify failure
            category, error_msg, error_hash = classify_failure(
                passed, stderr, exit_code, record.extracted_code
            )

            # Update record
            record.passed = passed
            record.failure_category = category
            record.error_message = error_msg
            record.error_hash = error_hash

            if passed:
                pass_count += 1
            category_counts[category] = category_counts.get(category, 0) + 1

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"  {i + 1}/{len(records)} ({elapsed:.1f}s, {pass_count} passed)")

        # Write updated records (atomic: write to tmp, then rename)
        tmp_path = shard_file.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            for record in records:
                f.write(record.to_json_line() + "\n")
        tmp_path.rename(shard_file)

        elapsed = time.time() - t0
        pct = 100 * pass_count / len(records) if records else 0
        print(f"  Done in {elapsed:.1f}s: {pass_count}/{len(records)} passed ({pct:.1f}%)")
        print(f"  Categories: {dict(sorted(category_counts.items()))}")

        total_records += len(records)
        total_passed += pass_count

    # --- Summary + W&B ---
    pct = 100 * total_passed / total_records if total_records else 0
    print(f"\nTotal: {total_passed}/{total_records} passed ({pct:.1f}%)")

    wandb.log({
        "total_records": total_records,
        "total_passed": total_passed,
        "pass_rate": pct,
    })
    wandb.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
