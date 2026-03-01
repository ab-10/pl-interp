"""Load and normalize HumanEval dataset for the experiment pipeline."""

from datasets import load_dataset


def load_humaneval() -> list[dict]:
    """Load HumanEval dataset, normalize each task to a standard dict format.

    Returns a list of dicts with keys:
        task_id, dataset, prompt, entry_point, test, canonical_solution
    """
    ds = load_dataset("openai_humaneval", split="test")
    tasks = []
    for i, row in enumerate(ds):
        tasks.append({
            "task_id": f"humaneval_{i}",
            "dataset": "humaneval",
            "prompt": row["prompt"],  # docstring + signature from the dataset
            "entry_point": row["entry_point"],
            "test": row["test"],
            "canonical_solution": row["canonical_solution"],
        })
    return tasks
