"""Batch generation with vLLM. Produces outputs with exact token IDs for downstream teacher-forcing."""

from vllm import LLM, SamplingParams

from experiments import config


class VLLMRunner:
    """Wraps vLLM's LLM for batch code generation on a single GPU."""

    def __init__(
        self,
        model_id: str = config.MODEL_ID,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        self.llm = LLM(model=model_id, gpu_memory_utilization=gpu_memory_utilization)

    def generate_batch(
        self,
        prompts: list[str],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
    ) -> list[dict]:
        """Generate completions for a batch of prompts.

        Returns list of {"text": str, "token_ids": list[int]}.
        Token IDs come directly from vLLM output — never re-tokenized.
        """
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        outputs = self.llm.generate(prompts, params)
        results = []
        for output in outputs:
            completion = output.outputs[0]
            results.append({
                "text": completion.text,
                "token_ids": list(completion.token_ids),
            })
        return results

    def generate_retry(
        self,
        prompts: list[str],
        max_tokens: int,
        seed: int,
    ) -> list[dict]:
        """Greedy retry for extraction failures (temp=0.0)."""
        return self.generate_batch(
            prompts=prompts,
            temperature=config.GREEDY_RETRY_TEMP,
            top_p=1.0,
            max_tokens=max_tokens,
            seed=seed,
        )
