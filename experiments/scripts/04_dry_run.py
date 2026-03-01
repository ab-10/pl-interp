"""Pre-flight dry run: exercises the real pipeline with a tiny subset before overnight run.

Tests 2 HumanEval + 2 MBPP tasks, 1 variant (baseline), 1 seed.
Runs all 3 stages sequentially with GPU memory management between vLLM and HF.

What this catches that 00_sanity_check doesn't:
  - MBPP loading, prompt building, and evaluation
  - The actual job-building / evaluation / capture logic from the orchestration modules
  - JSONL serialization round-trip through all 3 stages
  - Activation storage for both datasets

Usage:
  python3 -m experiments.scripts.04_dry_run
  python3 -m experiments.scripts.04_dry_run --model mistral-7b
"""

from __future__ import annotations

import argparse
import gc
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

import experiments.config as config
from transformers import AutoTokenizer

from experiments.datasets.load_humaneval import load_humaneval
from experiments.datasets.load_mbpp import load_mbpp
from experiments.evaluation.executor import execute_code
from experiments.evaluation.extractor import extract_code
from experiments.evaluation.judge import (
    build_humaneval_test_script,
    build_mbpp_test_script,
    classify_failure,
)
from experiments.generation.activation_capture import ActivationCapture
from experiments.generation.vllm_runner import VLLMRunner
from experiments.prompts.builder import build_humaneval_prompt, build_mbpp_prompt
from experiments.storage.activation_store import ActivationReader, ActivationWriter
from experiments.storage.schema import make_generation_record, read_records, write_records


def _print_result(label: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-flight dry run.")
    config.add_model_arg(parser)
    args = parser.parse_args()

    config.set_model(args.model)

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available.")
        return 1

    # Patch config to use temp dir and tiny subset
    _tmp_dir = Path(tempfile.mkdtemp(prefix="dry_run_"))
    config.SCRATCH_DIR = _tmp_dir
    config.GENERATIONS_DIR = _tmp_dir / "generations"
    config.ACTIVATIONS_DIR = _tmp_dir / "activations"
    config.VARIANT_IDS = ["baseline"]
    config.NUM_RUNS = 1

    all_ok = True
    t_start = time.time()

    print(f"Model: {config.MODEL_NAME} ({config.MODEL_ID})")
    print(f"Dry run temp dir: {_tmp_dir}")
    print(f"Config: 1 variant (baseline), 1 run (seed {config.BASE_SEED})\n")

    # =====================================================================
    # Load datasets
    # =====================================================================
    print("=== Loading datasets ===")
    humaneval_tasks = load_humaneval()[:2]
    mbpp_tasks = load_mbpp()[:2]
    all_tasks = humaneval_tasks + mbpp_tasks

    _print_result("HumanEval loaded", len(humaneval_tasks) == 2, f"{len(humaneval_tasks)} tasks")
    _print_result("MBPP loaded", len(mbpp_tasks) == 2, f"{len(mbpp_tasks)} tasks")

    for t in mbpp_tasks:
        has_keys = all(k in t for k in ["task_id", "dataset", "prompt", "function_name", "test_list"])
        _print_result(f"MBPP {t['task_id']} has expected keys", has_keys)
        if not has_keys:
            all_ok = False

    print(f"  Tasks: {[t['task_id'] for t in all_tasks]}")

    # =====================================================================
    # Stage 1: Generate (vLLM) — mirrors 01_generate.py logic
    # =====================================================================
    print("\n=== Stage 1: Generation (vLLM) ===")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)

    # Build jobs (same as 01_generate._build_jobs)
    jobs = []
    for task in all_tasks:
        for variant_id in config.VARIANT_IDS:
            if task["dataset"] == "humaneval":
                user_content = build_humaneval_prompt(task, variant_id)
                func_name = task["entry_point"]
            else:
                user_content = build_mbpp_prompt(task, variant_id)
                func_name = task["function_name"]

            messages = [{"role": "user", "content": user_content}]
            prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            full_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)

            for run_id in range(config.NUM_RUNS):
                jobs.append({
                    "task": task,
                    "variant_id": variant_id,
                    "run_id": run_id,
                    "seed": config.BASE_SEED + run_id,
                    "user_content": user_content,
                    "full_prompt": full_prompt,
                    "prompt_tokens": len(prompt_ids),
                    "func_name": func_name,
                })

    print(f"  Jobs: {len(jobs)} (4 tasks x 1 variant x 1 run)")

    print("  Initializing vLLM...")
    runner = VLLMRunner(gpu_memory_utilization=0.65)

    records_with_func: list[tuple] = []
    for run_id in range(config.NUM_RUNS):
        run_jobs = [j for j in jobs if j["run_id"] == run_id]
        seed = config.BASE_SEED + run_id
        prompts = [j["full_prompt"] for j in run_jobs]

        print(f"  Run {run_id} (seed={seed}): generating {len(prompts)} outputs...")
        results = runner.generate_batch(
            prompts=prompts,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            max_tokens=config.MAX_NEW_TOKENS,
            seed=seed,
        )

        for job, result in zip(run_jobs, results):
            task = job["task"]
            record = make_generation_record(
                task_id=task["task_id"],
                dataset=task["dataset"],
                variant_id=job["variant_id"],
                run_id=job["run_id"],
                seed=job["seed"],
                prompt_text=job["user_content"],
                prompt_tokens=job["prompt_tokens"],
                generated_text=result["text"],
                gen_token_ids=result["token_ids"],
                generated_tokens=len(result["token_ids"]),
            )
            records_with_func.append((record, job["func_name"]))

    # Extract code
    for record, func_name in records_with_func:
        extracted, clean = extract_code(record.generated_text, func_name)
        record.extracted_code = extracted
        record.extraction_clean = clean

    gen_ok = all(len(rec.gen_token_ids) > 0 for rec, _ in records_with_func)
    _print_result("All 4 generations produced tokens", gen_ok)
    if not gen_ok:
        all_ok = False

    clean_count = sum(1 for rec, _ in records_with_func if rec.extraction_clean)
    _print_result(f"Extraction: {clean_count}/4 clean", clean_count > 0)

    for rec, _ in records_with_func:
        print(f"    {rec.task_id} ({rec.dataset}): {len(rec.gen_token_ids)} tokens, clean={rec.extraction_clean}")

    # Save JSONL (same as 01_generate.py)
    config.GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.GENERATIONS_DIR / "shard_0.jsonl"
    all_records = [rec for rec, _ in records_with_func]
    write_records(output_path, all_records)

    written_ok = output_path.exists() and output_path.stat().st_size > 0
    _print_result("JSONL shard written", written_ok)
    if not written_ok:
        all_ok = False

    # =====================================================================
    # Stage 2: Evaluate — mirrors 02_evaluate.py logic
    # =====================================================================
    print("\n=== Stage 2: Evaluation ===")

    humaneval_lookup = {t["task_id"]: t for t in load_humaneval()}
    mbpp_lookup = {t["task_id"]: t for t in load_mbpp()}

    records = read_records(output_path)
    read_ok = len(records) == 4
    _print_result("JSONL round-trip", read_ok, f"read {len(records)} records")
    if not read_ok:
        all_ok = False

    pass_count = 0
    for record in records:
        if record.dataset == "humaneval":
            task = humaneval_lookup[record.task_id]
            test_script = build_humaneval_test_script(
                record.extracted_code, task["test"], task["entry_point"]
            )
        else:
            task = mbpp_lookup[record.task_id]
            test_script = build_mbpp_test_script(
                record.extracted_code, task["test_list"], task["test_setup_code"]
            )

        passed, stderr, exit_code = execute_code(test_script, timeout=config.EXTRACTION_TIMEOUT)
        category, error_msg, error_hash = classify_failure(
            passed, stderr, exit_code, record.extracted_code
        )

        record.passed = passed
        record.failure_category = category
        record.error_message = error_msg
        record.error_hash = error_hash

        if passed:
            pass_count += 1
        print(f"    {record.task_id}: {category}")

    _print_result(f"Evaluation: {pass_count}/4 passed", True)

    # Atomic write (same as 02_evaluate.py)
    tmp_path = output_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        for record in records:
            f.write(record.to_json_line() + "\n")
    tmp_path.rename(output_path)

    eval_records = read_records(output_path)
    eval_ok = all(rec.failure_category != "" for rec in eval_records)
    _print_result("Evaluated records saved", eval_ok)
    if not eval_ok:
        all_ok = False

    # =====================================================================
    # Stage 3: Activation Capture — mirrors 03_capture_activations.py logic
    # =====================================================================
    print("\n=== Stage 3: Activation Capture ===")

    print("  Freeing vLLM...")
    del runner
    gc.collect()
    torch.cuda.empty_cache()

    print("  Loading HF model...")
    capture = ActivationCapture()

    config.ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
    act_path = config.ACTIVATIONS_DIR / "shard_0.npy"
    writer = ActivationWriter(act_path)

    records = read_records(output_path)
    activations = capture.capture_batch(records, batch_size=len(records))

    total_tokens = 0
    for record, acts in zip(records, activations):
        offset, length = writer.append(acts)
        record.activation_file = str(act_path)
        record.activation_offset = offset
        record.activation_length = length
        total_tokens += length

    act_ok = act_path.exists() and act_path.stat().st_size > 0
    _print_result(
        "Activations written",
        act_ok,
        f"{total_tokens} tokens, {act_path.stat().st_size / 1024 / 1024:.1f} MB",
    )
    if not act_ok:
        all_ok = False

    # Verify shapes via reader
    reader = ActivationReader(act_path)
    for record in records:
        act = reader.read(record.activation_offset, record.activation_length)
        expected_rows = len(record.gen_token_ids)
        shape_ok = act.shape == (expected_rows, config.MODEL_HIDDEN_DIM)
        _print_result(
            f"Activation shape {record.task_id}",
            shape_ok,
            f"got {act.shape}, expected ({expected_rows}, {config.MODEL_HIDDEN_DIM})",
        )
        if not shape_ok:
            all_ok = False

    # Atomic write (same as 03_capture_activations.py)
    tmp_path = output_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        for record in records:
            f.write(record.to_json_line() + "\n")
    tmp_path.rename(output_path)

    # =====================================================================
    # Final Verification
    # =====================================================================
    print("\n=== Final Verification ===")
    final_records = read_records(output_path)
    for rec in final_records:
        complete = (
            rec.task_id != ""
            and rec.failure_category != ""
            and rec.activation_file != ""
            and rec.activation_length > 0
            and len(rec.gen_token_ids) > 0
        )
        _print_result(
            f"{rec.task_id} ({rec.dataset})",
            complete,
            f"tokens={len(rec.gen_token_ids)}, category={rec.failure_category}, act_len={rec.activation_length}",
        )
        if not complete:
            all_ok = False

    # Cleanup
    shutil.rmtree(_tmp_dir, ignore_errors=True)
    elapsed = time.time() - t_start
    print(f"\nCompleted in {elapsed:.1f}s (temp dir cleaned up)")

    print("\n" + "=" * 60)
    if all_ok:
        print("DRY RUN PASSED — safe to start the full overnight run.")
    else:
        print("DRY RUN FAILED — fix issues before starting full run.")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
