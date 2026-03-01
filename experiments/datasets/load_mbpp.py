"""Load and normalize MBPP dataset for the experiment pipeline."""

import re

from datasets import load_dataset


def _extract_function_name(test_list: list[str]) -> str:
    """Extract function name from MBPP assertion strings.

    Parses assertions like 'assert func_name(args) == expected' to get func_name.
    """
    for test_str in test_list:
        match = re.search(r"assert\s+(\w+)\s*\(", test_str)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract function name from test_list: {test_list}")


def load_mbpp() -> list[dict]:
    """Load full MBPP dataset (all splits: train+test+validation+prompt = 974 tasks).

    Deduplicates by task_id in case of overlap between splits.

    Returns a list of dicts with keys:
        task_id, dataset, prompt, function_name, test_list, test_setup_code
    """
    seen_ids: set[int] = set()
    tasks = []

    for split in ["test", "train", "validation", "prompt"]:
        ds = load_dataset("mbpp", split=split)
        for row in ds:
            raw_id = row["task_id"]
            if raw_id in seen_ids:
                continue
            seen_ids.add(raw_id)

            task_id = f"mbpp_{raw_id}"
            test_list = row["test_list"]
            function_name = _extract_function_name(test_list)
            tasks.append({
                "task_id": task_id,
                "dataset": "mbpp",
                "prompt": row["text"],
                "function_name": function_name,
                "test_list": test_list,
                "test_setup_code": row.get("test_setup_code", "") or "",
            })

    return tasks
