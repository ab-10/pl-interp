"""HF teacher-forcing forward pass for multi-layer activation capture."""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments import config
from experiments.storage.schema import GenerationRecord


class ActivationCapture:
    """Captures activations at multiple layers via HuggingFace output_hidden_states."""

    def __init__(
        self,
        model_id: str = config.MODEL_ID,
        device: str = "cuda",
    ) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map=device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model.eval()

    def capture_batch(
        self,
        records: list[GenerationRecord],
        batch_size: int = 16,
    ) -> list[dict[int, np.ndarray]]:
        """Capture multi-layer activations for a list of generation records.

        For each record, constructs input_ids by concatenating prompt token IDs
        (from tokenizer) with stored gen_token_ids — never re-tokenizes decoded text.

        Returns list of dicts, one per record:
            {layer_num: np.ndarray of shape (num_gen_tokens, hidden_dim)}
        """
        all_activations: list[dict[int, np.ndarray]] = []

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            batch_activations = self._capture_single_batch(batch)
            all_activations.extend(batch_activations)

        return all_activations

    def _capture_single_batch(
        self,
        records: list[GenerationRecord],
    ) -> list[dict[int, np.ndarray]]:
        """Process a single batch of records through the model."""
        results: list[dict[int, np.ndarray]] = []

        for record in records:
            messages = [{"role": "user", "content": record.prompt_text}]
            prompt_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            prompt_length = len(prompt_ids)
            input_ids = prompt_ids + record.gen_token_ids
            input_tensor = torch.tensor(
                [input_ids],
                dtype=torch.long,
                device=self.model.device,
            )

            with torch.no_grad():
                outputs = self.model(input_tensor, output_hidden_states=True)

            # Extract activations at each capture layer
            layer_acts = {}
            for layer_num, hs_idx in zip(
                config.CAPTURE_LAYERS, config.HIDDEN_STATES_INDICES
            ):
                layer_hidden = outputs.hidden_states[hs_idx]
                gen_activations = layer_hidden[:, prompt_length:, :]
                act_np = gen_activations.squeeze(0).cpu().numpy().astype(np.float16)
                layer_acts[layer_num] = act_np

            results.append(layer_acts)

        return results
