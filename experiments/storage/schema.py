"""GenerationRecord dataclass and JSONL serialization. Every pipeline stage reads/writes these."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class GenerationRecord:
    # Identity
    task_id: str              # "humaneval_023" or "mbpp_412"
    dataset: str              # "humaneval" or "mbpp"
    variant_id: str           # "baseline", "typed", "invariants", etc.
    run_id: int               # 0, 1, 2
    seed: int                 # BASE_SEED + run_id

    # Prompt
    prompt_text: str          # Full assembled prompt
    prompt_tokens: int        # Token count

    # Generation
    generated_text: str       # Raw model output
    extracted_code: str       # Regex-extracted code (empty if extraction failed)
    gen_token_ids: list[int]  # Token IDs of generated sequence (from vLLM, never re-tokenized)
    generated_tokens: int     # Token count

    # Extraction
    extraction_clean: bool    # Whether first extraction succeeded
    extraction_retried: bool  # Whether a greedy retry was attempted
    retry_succeeded: bool     # Whether retry extraction succeeded

    # Evaluation
    passed: bool              # All tests passed
    failure_category: str     # pass|wrong_answer|syntax_error|type_error|runtime_error|timeout|extraction_fail
    error_message: str        # Raw error message (if any)
    error_hash: str           # Hash of error message for clustering

    # Activation metadata (per-layer, stored separately in mmap files)
    # Keys are layer numbers (as strings in JSON), values are {file, offset, length}
    activation_layers: dict   # {layer_num: {"file": str, "offset": int, "length": int}}

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json_line(cls, line: str) -> GenerationRecord:
        return cls(**json.loads(line))


def compute_error_hash(error_message: str) -> str:
    """Stable hash for clustering identical errors across records."""
    if not error_message:
        return ""
    return hashlib.sha256(error_message.encode()).hexdigest()[:16]


def write_records(path: Path, records: list[GenerationRecord]) -> None:
    """Append records to a JSONL file. Creates file if it doesn't exist."""
    with open(path, "a") as f:
        for record in records:
            f.write(record.to_json_line() + "\n")


def read_records(path: Path) -> list[GenerationRecord]:
    """Read all records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(GenerationRecord.from_json_line(line))
    return records


# --- Partial record constructors ---
# Pipeline stages fill fields incrementally. These provide defaults for
# fields not yet populated so records can be serialized between stages.

def make_generation_record(
    task_id: str,
    dataset: str,
    variant_id: str,
    run_id: int,
    seed: int,
    prompt_text: str,
    prompt_tokens: int,
    generated_text: str = "",
    extracted_code: str = "",
    gen_token_ids: list[int] | None = None,
    generated_tokens: int = 0,
    extraction_clean: bool = False,
    extraction_retried: bool = False,
    retry_succeeded: bool = False,
    passed: bool = False,
    failure_category: str = "",
    error_message: str = "",
    error_hash: str = "",
    activation_layers: dict | None = None,
) -> GenerationRecord:
    """Create a GenerationRecord with sensible defaults for unfilled fields."""
    return GenerationRecord(
        task_id=task_id,
        dataset=dataset,
        variant_id=variant_id,
        run_id=run_id,
        seed=seed,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
        generated_text=generated_text,
        extracted_code=extracted_code,
        gen_token_ids=gen_token_ids if gen_token_ids is not None else [],
        generated_tokens=generated_tokens,
        extraction_clean=extraction_clean,
        extraction_retried=extraction_retried,
        retry_succeeded=retry_succeeded,
        passed=passed,
        failure_category=failure_category,
        error_message=error_message,
        error_hash=error_hash,
        activation_layers=activation_layers if activation_layers is not None else {},
    )
