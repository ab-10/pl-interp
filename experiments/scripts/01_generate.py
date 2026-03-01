"""Stage 1: Generate all code outputs with vLLM. Sharded by task_id across GPUs.

Reads: HumanEval + MBPP datasets (downloaded automatically)
Writes: /scratch/generations/shard_{N}.jsonl

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m experiments.scripts.01_generate --shard 0
  CUDA_VISIBLE_DEVICES=1 python -m experiments.scripts.01_generate --shard 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

from experiments import config
from experiments.datasets.load_humaneval import load_humaneval
from experiments.datasets.load_mbpp import load_mbpp
from experiments.evaluation.extractor import extract_code
from experiments.generation.vllm_runner import VLLMRunner
from experiments.prompts.builder import build_humaneval_prompt, build_mbpp_prompt
from experiments.storage.schema import make_generation_record, write_records


def _build_jobs(tasks: list[dict], tokenizer) -> list[dict]:
    """Build all (task, variant, run) combinations with chat-templated prompts."""
    jobs = []
    for task in tasks:
        for variant_id in config.VARIANT_IDS:
            if task["dataset"] == "humaneval":
                user_content = build_humaneval_prompt(task, variant_id)
                func_name = task["entry_point"]
            else:
                user_content = build_mbpp_prompt(task, variant_id)
                func_name = task["function_name"]

            # Apply chat template for vLLM prompt
            messages = [{"role": "user", "content": user_content}]
            prompt_token_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            full_prompt = tokenizer.decode(
                prompt_token_ids, skip_special_tokens=False
            )
            prompt_tokens = len(prompt_token_ids)

            for run_id in range(config.NUM_RUNS):
                jobs.append({
                    "task": task,
                    "variant_id": variant_id,
                    "run_id": run_id,
                    "seed": config.BASE_SEED + run_id,
                    "user_content": user_content,
                    "full_prompt": full_prompt,
                    "prompt_tokens": prompt_tokens,
                    "func_name": func_name,
                })
    return jobs


def _apply_chat_template(tokenizer, user_content: str) -> str:
    """Apply Mistral chat template to user message content."""
    messages = [{"role": "user", "content": user_content}]
    token_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 1: Generate code outputs with vLLM, sharded by task_id.",
    )
    parser.add_argument(
        "--shard", type=int, required=True,
        help="GPU shard index (0-based)",
    )
    parser.add_argument(
        "--num-shards", type=int, default=config.NUM_GPUS,
        help=f"Total number of GPU shards (default: {config.NUM_GPUS})",
    )
    args = parser.parse_args()

    config.GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.GENERATIONS_DIR / f"shard_{args.shard}.jsonl"

    # --- Load tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)

    # --- Load datasets ---
    print("Loading datasets...")
    all_tasks = load_humaneval() + load_mbpp()
    print(f"Total tasks: {len(all_tasks)}")

    # Shard by task index
    my_tasks = all_tasks[args.shard :: args.num_shards]
    print(f"Shard {args.shard}/{args.num_shards}: {len(my_tasks)} tasks")

    # --- Build all jobs ---
    jobs = _build_jobs(my_tasks, tokenizer)
    print(f"Total jobs (tasks x variants x runs): {len(jobs)}")

    # --- Initialize vLLM ---
    print("Initializing vLLM...")
    runner = VLLMRunner()

    # --- Generate per run_id (each run has a different seed) ---
    records_with_func = []  # list of (GenerationRecord, func_name)

    for run_id in range(config.NUM_RUNS):
        run_jobs = [j for j in jobs if j["run_id"] == run_id]
        seed = config.BASE_SEED + run_id

        prompts = [j["full_prompt"] for j in run_jobs]
        print(f"\nRun {run_id} (seed={seed}): generating {len(prompts)} outputs...")
        t0 = time.time()

        results = runner.generate_batch(
            prompts=prompts,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            max_tokens=config.MAX_NEW_TOKENS,
            seed=seed,
        )
        elapsed = time.time() - t0
        print(f"  Generated in {elapsed:.1f}s ({len(prompts)/elapsed:.1f} prompts/s)")

        for job, result in zip(run_jobs, results):
            task = job["task"]
            record = make_generation_record(
                task_id=task["task_id"],
                dataset=task["dataset"],
                variant_id=job["variant_id"],
                run_id=job["run_id"],
                seed=job["seed"],
                prompt_text=job["user_content"],  # store user message content
                prompt_tokens=job["prompt_tokens"],
                generated_text=result["text"],
                gen_token_ids=result["token_ids"],
                generated_tokens=len(result["token_ids"]),
            )
            records_with_func.append((record, job["func_name"]))

    # --- Extract code ---
    print(f"\nExtracting code from {len(records_with_func)} generations...")
    extraction_failures = []
    for record, func_name in records_with_func:
        extracted, clean = extract_code(record.generated_text, func_name)
        record.extracted_code = extracted
        record.extraction_clean = clean
        if not clean:
            extraction_failures.append((record, func_name))

    clean_count = len(records_with_func) - len(extraction_failures)
    print(f"  {clean_count} clean, {len(extraction_failures)} failures")

    # --- Retry extraction failures with greedy ---
    if extraction_failures:
        print(f"\nRetrying {len(extraction_failures)} failures at temp=0.0...")
        retry_prompts = [
            _apply_chat_template(tokenizer, rec.prompt_text)
            for rec, _ in extraction_failures
        ]
        retry_results = runner.generate_retry(
            prompts=retry_prompts,
            max_tokens=config.MAX_NEW_TOKENS,
            seed=config.BASE_SEED,
        )

        retried_ok = 0
        for (record, func_name), result in zip(extraction_failures, retry_results):
            record.extraction_retried = True
            extracted, clean = extract_code(result["text"], func_name)
            if clean:
                # Replace generation with retry output
                record.generated_text = result["text"]
                record.gen_token_ids = result["token_ids"]
                record.generated_tokens = len(result["token_ids"])
                record.extracted_code = extracted
                record.extraction_clean = True
                record.retry_succeeded = True
                retried_ok += 1

        still_failed = len(extraction_failures) - retried_ok
        print(f"  Retry: {retried_ok} recovered, {still_failed} still failed")

    # --- Save records ---
    all_records = [rec for rec, _ in records_with_func]
    print(f"\nSaving {len(all_records)} records to {output_path}...")
    # Truncate if file exists (fresh write, not append)
    if output_path.exists():
        output_path.unlink()
    write_records(output_path, all_records)

    # --- Summary ---
    clean_final = sum(1 for r in all_records if r.extraction_clean)
    retried = sum(1 for r in all_records if r.extraction_retried)
    retry_ok = sum(1 for r in all_records if r.retry_succeeded)
    print(f"\nSummary:")
    print(f"  Total records: {len(all_records)}")
    print(f"  Clean extraction: {clean_final}")
    print(f"  Retried: {retried} (succeeded: {retry_ok})")
    print(f"  Extraction failures: {len(all_records) - clean_final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
