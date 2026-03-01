"""Sanity checks and end-to-end tests. Run before any full pipeline stage to validate
indexing, shapes, determinism, steering hooks, and token round-trips.

Modes:
  default:    5 sanity checks + micro E2E (HF generate stand-in)
  --skip-e2e: 5 sanity checks only
  --full-e2e: 5 sanity checks + full pipeline E2E (vLLM + activation capture)
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments import config
from experiments.datasets.load_humaneval import load_humaneval
from experiments.evaluation.extractor import extract_code
from experiments.evaluation.executor import execute_code
from experiments.evaluation.judge import build_humaneval_test_script, classify_failure
from experiments.prompts.builder import build_humaneval_prompt
from experiments.steering.hook import attach_steering_hook, make_steering_hook
from experiments.storage.activation_store import ActivationReader, ActivationWriter
from experiments.storage.schema import GenerationRecord, make_generation_record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_result(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")


def _load_model_and_tokenizer() -> tuple:
    """Load the HF model and tokenizer once for all checks."""
    print(f"Loading model {config.MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ---------------------------------------------------------------------------
# Check 1: Hidden state indexing
# ---------------------------------------------------------------------------

def check_hidden_state_indexing(model, tokenizer) -> bool:
    """Verify outputs.hidden_states[17] == hooked layer-16 output (exact equality)."""
    print("\nCheck 1: Hidden state indexing")

    prompts = ["def hello():\n    return", "for i in range(10):\n    print(i)"]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=32).to(model.device)

    hooked_output = {}

    def capture_hook(module, inp, out):
        # Newer transformers: bare tensor. Older: tuple (hidden_states, ...).
        tensor = out[0] if isinstance(out, tuple) else out
        hooked_output["tensor"] = tensor.detach().clone()

    handle = model.model.layers[config.CAPTURE_LAYER].register_forward_hook(capture_hook)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    handle.remove()

    hs_from_output = outputs.hidden_states[config.HIDDEN_STATES_INDEX]
    hs_from_hook = hooked_output["tensor"]

    match = torch.equal(hs_from_output, hs_from_hook)
    _print_result(
        "hidden_states[17] == hooked layer-16 output",
        match,
        f"shapes: output={hs_from_output.shape}, hook={hs_from_hook.shape}",
    )
    return match


# ---------------------------------------------------------------------------
# Check 2: Shape validation
# ---------------------------------------------------------------------------

def check_shape_validation(model, tokenizer) -> bool:
    """Confirm hidden_states shapes and count."""
    print("\nCheck 2: Shape validation")

    prompt = "def hello():\n    return 'world'"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=32).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hs = outputs.hidden_states
    num_states = len(hs)
    expected_num = config.MODEL_NUM_LAYERS + 1  # 1 embedding + 32 layers = 33

    count_ok = num_states == expected_num
    _print_result(
        f"len(hidden_states) == {expected_num}",
        count_ok,
        f"got {num_states}",
    )

    layer_hs = hs[config.HIDDEN_STATES_INDEX]
    batch, seq_len, hidden_dim = layer_hs.shape
    shape_ok = hidden_dim == config.MODEL_HIDDEN_DIM
    _print_result(
        f"hidden_states[{config.HIDDEN_STATES_INDEX}].shape[-1] == {config.MODEL_HIDDEN_DIM}",
        shape_ok,
        f"shape={layer_hs.shape}",
    )

    return count_ok and shape_ok


# ---------------------------------------------------------------------------
# Check 3: Teacher-forcing determinism
# ---------------------------------------------------------------------------

def check_teacher_forcing_determinism(model, tokenizer) -> bool:
    """Run same tokens twice with eval() + manual_seed — activations must be bitwise identical."""
    print("\nCheck 3: Teacher-forcing determinism")

    prompt = "def factorial(n):\n    if n <= 1:\n        return 1"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=32).to(model.device)

    model.eval()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    with torch.no_grad():
        out1 = model(**inputs, output_hidden_states=True)
    act1 = out1.hidden_states[config.HIDDEN_STATES_INDEX].clone()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    with torch.no_grad():
        out2 = model(**inputs, output_hidden_states=True)
    act2 = out2.hidden_states[config.HIDDEN_STATES_INDEX].clone()

    match = torch.equal(act1, act2)
    _print_result("teacher-forcing bitwise determinism", match)
    return match


# ---------------------------------------------------------------------------
# Check 4: Steering hook — tuple handling + decode-only gating
# ---------------------------------------------------------------------------

def check_steering_hook(model, tokenizer) -> bool:
    """Validate steering hook: alpha=0 no-op, large alpha differs, decode-only gating."""
    print("\nCheck 4: Steering hook — tuple handling + decode-only gating")

    prompt = "Write a Python function to compute factorial."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Baseline: unhooked generation
    torch.manual_seed(42)
    with torch.no_grad():
        baseline_ids = model.generate(**inputs, **gen_kwargs)

    # Alpha=0: should be identical to baseline
    direction = torch.randn(config.MODEL_HIDDEN_DIM, device=model.device, dtype=torch.float16)
    handle_zero = attach_steering_hook(model, config.CAPTURE_LAYER, direction, alpha=0.0)
    torch.manual_seed(42)
    with torch.no_grad():
        alpha0_ids = model.generate(**inputs, **gen_kwargs)
    handle_zero.remove()

    zero_match = torch.equal(baseline_ids, alpha0_ids)
    _print_result("alpha=0 output == unhooked output", zero_match)

    # Large alpha: output should differ
    handle_large = attach_steering_hook(model, config.CAPTURE_LAYER, direction, alpha=10.0)

    # Track hook calls: wrap the layer's hook to count
    fire_count = {"total": 0, "injected": 0}
    # Remove the handle we just attached and re-attach with counting
    handle_large.remove()

    original_hook_fn = make_steering_hook(direction, alpha=10.0)

    def counting_hook(module, inp, out):
        fire_count["total"] += 1
        hidden_states = out[0] if isinstance(out, tuple) else out
        if hidden_states.shape[1] == 1:
            fire_count["injected"] += 1
        return original_hook_fn(module, inp, out)

    handle_counting = model.model.layers[config.CAPTURE_LAYER].register_forward_hook(counting_hook)

    torch.manual_seed(42)
    with torch.no_grad():
        large_alpha_ids = model.generate(**inputs, **gen_kwargs)
    handle_counting.remove()

    large_differs = not torch.equal(baseline_ids, large_alpha_ids)
    _print_result("alpha=10.0 output != unhooked output", large_differs)

    # Decode-only gating: prefill processes full prompt and produces 1st new token,
    # then (N-1) decode steps produce tokens 2..N. Total = N calls, injected = N-1.
    num_new_tokens = large_alpha_ids.shape[1] - inputs["input_ids"].shape[1]
    expected_total = num_new_tokens       # 1 prefill + (N-1) decode = N
    expected_injected = num_new_tokens - 1  # only decode steps (shape[1]==1)

    gating_ok = (fire_count["injected"] == expected_injected and
                 fire_count["total"] == expected_total)
    _print_result(
        "decode-only gating",
        gating_ok,
        f"total_fires={fire_count['total']} (expected {expected_total}), "
        f"injected={fire_count['injected']} (expected {expected_injected})",
    )

    return zero_match and large_differs and gating_ok


# ---------------------------------------------------------------------------
# Check 5: Token ID round-trip
# ---------------------------------------------------------------------------

def check_token_round_trip(model, tokenizer) -> bool:
    """Verify gen_token_ids can be used directly for teacher-forcing replay.

    Our pipeline never re-tokenizes text — we store raw token IDs from generation
    and feed them directly to the HF model. This check validates that:
    1. All gen_token_ids are valid vocabulary indices
    2. Concatenating prompt_ids + gen_token_ids produces a valid model input
    3. The model can run a forward pass on the concatenated sequence
    """
    print("\nCheck 5: Token ID validity + teacher-forcing replay")

    prompt = "def add(a, b):\n    return a + b\n\ndef multiply"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0, prompt_len:].tolist()

    # Check 1: All token IDs are within vocabulary range
    vocab_size = tokenizer.vocab_size
    all_valid = all(0 <= tid < vocab_size for tid in gen_ids)
    _print_result(
        "all gen_token_ids within vocab range",
        all_valid,
        f"{len(gen_ids)} tokens, vocab_size={vocab_size}",
    )

    # Check 2: Decoded text is non-empty and readable
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
    has_content = len(decoded.strip()) > 0
    _print_result(
        "decoded text is non-empty",
        has_content,
        f"{len(decoded)} chars",
    )

    # Check 3: Teacher-forcing replay works — concatenate prompt + gen IDs,
    # feed to model, get valid output (this is exactly what activation_capture does)
    prompt_ids = inputs["input_ids"][0].tolist()
    full_ids = prompt_ids + gen_ids
    full_tensor = torch.tensor([full_ids], dtype=torch.long, device=model.device)

    with torch.no_grad():
        replay_out = model(full_tensor, output_hidden_states=True)

    layer16 = replay_out.hidden_states[config.HIDDEN_STATES_INDEX]
    replay_ok = (
        layer16.shape[0] == 1
        and layer16.shape[1] == len(full_ids)
        and layer16.shape[2] == config.MODEL_HIDDEN_DIM
    )
    _print_result(
        "teacher-forcing replay produces valid activations",
        replay_ok,
        f"shape={layer16.shape}, expected=(1, {len(full_ids)}, {config.MODEL_HIDDEN_DIM})",
    )

    return all_valid and has_content and replay_ok


# ---------------------------------------------------------------------------
# Micro end-to-end
# ---------------------------------------------------------------------------

def micro_end_to_end(model, tokenizer) -> bool:
    """Load 2 HumanEval tasks, generate, extract, execute, classify, build record."""
    print("\nMicro end-to-end test")

    # Load 2 HumanEval tasks
    print("  Loading HumanEval ...")
    tasks = load_humaneval()[:2]
    print(f"  Loaded {len(tasks)} tasks: {[t['task_id'] for t in tasks]}")

    all_ok = True

    for task in tasks:
        task_id = task["task_id"]
        print(f"\n  --- {task_id} ({task['entry_point']}) ---")

        # Build prompt
        prompt_text = build_humaneval_prompt(task, "baseline")
        print(f"  Prompt length: {len(prompt_text)} chars")

        # Generate with HF (stand-in for vLLM in smoke test)
        messages = [{"role": "user", "content": prompt_text}]
        chat_input = tokenizer.apply_chat_template(messages, return_tensors="pt",
                                                   add_generation_prompt=True).to(model.device)
        prompt_tokens = chat_input.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                chat_input,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=True,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_token_ids = output_ids[0, prompt_tokens:].tolist()
        generated_text = tokenizer.decode(gen_token_ids, skip_special_tokens=True)
        print(f"  Generated {len(gen_token_ids)} tokens")

        # Extract code
        extracted_code, extraction_clean = extract_code(generated_text, task["entry_point"])
        print(f"  Extraction clean: {extraction_clean}")
        if extracted_code:
            preview = extracted_code[:80].replace("\n", "\\n")
            print(f"  Code preview: {preview}...")

        # Build test script and execute
        test_script = build_humaneval_test_script(extracted_code, task["test"], task["entry_point"])
        passed, stderr, exit_code = execute_code(test_script, timeout=config.EXTRACTION_TIMEOUT)

        # Classify failure
        failure_category, error_message, error_hash = classify_failure(
            passed, stderr, exit_code, extracted_code
        )
        print(f"  Result: {failure_category}" + (f" — {error_message[:80]}" if error_message else ""))

        # Build a valid GenerationRecord
        record = make_generation_record(
            task_id=task_id,
            dataset="humaneval",
            variant_id="baseline",
            run_id=0,
            seed=config.BASE_SEED,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
            generated_text=generated_text,
            extracted_code=extracted_code,
            gen_token_ids=gen_token_ids,
            generated_tokens=len(gen_token_ids),
            extraction_clean=extraction_clean,
            passed=passed,
            failure_category=failure_category,
            error_message=error_message,
            error_hash=error_hash,
        )

        # Validate record can round-trip through JSON
        json_line = record.to_json_line()
        recovered = GenerationRecord.from_json_line(json_line)
        record_ok = (
            recovered.task_id == record.task_id
            and recovered.failure_category == record.failure_category
            and recovered.gen_token_ids == record.gen_token_ids
        )
        _print_result(f"GenerationRecord round-trip for {task_id}", record_ok)
        if not record_ok:
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Full end-to-end (actual vLLM + activation capture)
# ---------------------------------------------------------------------------

def full_end_to_end(model, tokenizer) -> bool:
    """Complete pipeline test: vLLM generate → extract → execute → capture activations.

    Uses a temp directory instead of /scratch. Tests the real vLLM runner and
    activation capture/storage modules.
    """
    from experiments.generation.activation_capture import ActivationCapture
    from experiments.generation.vllm_runner import VLLMRunner

    print("\nFull end-to-end test (vLLM + activation capture)")

    tmp_dir = Path(tempfile.mkdtemp(prefix="smoke_e2e_"))
    print(f"  Temp dir: {tmp_dir}")

    try:
        # 1. Load 2 HumanEval tasks
        print("  Loading HumanEval...")
        tasks = load_humaneval()[:2]
        print(f"  Tasks: {[t['task_id'] for t in tasks]}")

        # 2. Build prompts with chat template
        jobs = []
        for task in tasks:
            user_content = build_humaneval_prompt(task, "baseline")
            messages = [{"role": "user", "content": user_content}]
            prompt_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            full_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
            jobs.append({
                "task": task,
                "user_content": user_content,
                "full_prompt": full_prompt,
                "prompt_tokens": len(prompt_ids),
            })

        # 3. Generate with actual vLLM
        print("  Initializing vLLM...")
        runner = VLLMRunner()
        prompts = [j["full_prompt"] for j in jobs]
        print(f"  Generating {len(prompts)} outputs with vLLM...")
        results = runner.generate_batch(
            prompts=prompts,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            max_tokens=config.MAX_NEW_TOKENS,
            seed=config.BASE_SEED,
        )

        # 4. Build records, extract, execute, classify
        records = []
        all_ok = True
        for job, result in zip(jobs, results):
            task = job["task"]
            generated_text = result["text"]
            gen_token_ids = result["token_ids"]

            extracted, clean = extract_code(generated_text, task["entry_point"])
            test_script = build_humaneval_test_script(
                extracted, task["test"], task["entry_point"]
            )
            passed, stderr, exit_code = execute_code(
                test_script, timeout=config.EXTRACTION_TIMEOUT
            )
            category, error_msg, error_hash = classify_failure(
                passed, stderr, exit_code, extracted
            )

            record = make_generation_record(
                task_id=task["task_id"],
                dataset="humaneval",
                variant_id="baseline",
                run_id=0,
                seed=config.BASE_SEED,
                prompt_text=job["user_content"],
                prompt_tokens=job["prompt_tokens"],
                generated_text=generated_text,
                extracted_code=extracted,
                gen_token_ids=gen_token_ids,
                generated_tokens=len(gen_token_ids),
                extraction_clean=clean,
                passed=passed,
                failure_category=category,
                error_message=error_msg,
                error_hash=error_hash,
            )
            records.append(record)
            print(f"  {task['task_id']}: {category} (extraction_clean={clean})")

        # 5. Capture activations using the ALREADY-LOADED HF model
        print("  Capturing activations (reusing HF model)...")
        capture = ActivationCapture.__new__(ActivationCapture)
        capture.model = model
        capture.tokenizer = tokenizer

        act_path = tmp_dir / "activations.npy"
        writer = ActivationWriter(act_path)

        activations_list = capture.capture_batch(records, batch_size=2)
        for record, acts in zip(records, activations_list):
            offset, length = writer.append(acts)
            record.activation_file = str(act_path)
            record.activation_offset = offset
            record.activation_length = length

        # 6. Read back activations and verify
        print("  Verifying activation storage round-trip...")
        reader = ActivationReader(act_path)
        for record in records:
            stored = reader.read(record.activation_offset, record.activation_length)
            shape_ok = (
                stored.shape[0] == record.activation_length
                and stored.shape[1] == config.MODEL_HIDDEN_DIM
            )
            _print_result(
                f"Activations for {record.task_id}",
                shape_ok,
                f"shape={stored.shape}, expected=({record.activation_length}, {config.MODEL_HIDDEN_DIM})",
            )
            if not shape_ok:
                all_ok = False

        # 7. Verify GenerationRecord JSON round-trip
        for record in records:
            json_line = record.to_json_line()
            recovered = GenerationRecord.from_json_line(json_line)
            fields_ok = (
                recovered.task_id == record.task_id
                and recovered.gen_token_ids == record.gen_token_ids
                and recovered.activation_offset == record.activation_offset
                and recovered.activation_length == record.activation_length
            )
            _print_result(f"Record round-trip for {record.task_id}", fields_ok)
            if not fields_ok:
                all_ok = False

        # 8. Verify vLLM token IDs match HF tokenizer round-trip
        print("  Verifying vLLM ↔ HF token ID compatibility...")
        for record in records:
            decoded = tokenizer.decode(record.gen_token_ids, skip_special_tokens=False)
            re_encoded = tokenizer.encode(decoded, add_special_tokens=False)
            match = record.gen_token_ids == re_encoded
            _print_result(
                f"vLLM token round-trip for {record.task_id}",
                match,
                f"{len(record.gen_token_ids)} tokens",
            )
            if not match:
                all_ok = False

        return all_ok

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity checks and end-to-end tests. "
            "Run on GPU before any full pipeline stage."
        ),
    )
    e2e_group = parser.add_mutually_exclusive_group()
    e2e_group.add_argument(
        "--skip-e2e",
        action="store_true",
        help="Skip all E2E tests (run only the 5 sanity checks).",
    )
    e2e_group.add_argument(
        "--full-e2e",
        action="store_true",
        help="Run full pipeline E2E with actual vLLM + activation capture.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available. Sanity checks require a GPU.")
        return 1

    model, tokenizer = _load_model_and_tokenizer()

    checks = [
        ("Check 1: Hidden state indexing", check_hidden_state_indexing),
        ("Check 2: Shape validation", check_shape_validation),
        ("Check 3: Teacher-forcing determinism", check_teacher_forcing_determinism),
        ("Check 4: Steering hook", check_steering_hook),
        ("Check 5: Token ID round-trip", check_token_round_trip),
    ]

    results = {}
    for name, fn in checks:
        try:
            results[name] = fn(model, tokenizer)
        except Exception:
            print(f"\n{name} — EXCEPTION:")
            traceback.print_exc()
            results[name] = False

        if not results[name]:
            print(f"\nAborting: {name} failed. Fix before continuing.")
            return 1

    if not args.skip_e2e:
        # Always run micro E2E (uses HF generate, no vLLM needed)
        try:
            e2e_ok = micro_end_to_end(model, tokenizer)
            results["Micro end-to-end"] = e2e_ok
        except Exception:
            print("\nMicro end-to-end — EXCEPTION:")
            traceback.print_exc()
            results["Micro end-to-end"] = False

    if args.full_e2e:
        # Full pipeline E2E with actual vLLM + activation capture
        try:
            full_ok = full_end_to_end(model, tokenizer)
            results["Full end-to-end"] = full_ok
        except Exception:
            print("\nFull end-to-end — EXCEPTION:")
            traceback.print_exc()
            results["Full end-to-end"] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll checks passed. Safe to run pipeline.")
        return 0
    else:
        print("\nSome checks failed. Do NOT run pipeline until fixed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
