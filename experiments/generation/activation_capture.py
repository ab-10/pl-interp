"""HF teacher-forcing forward pass for activation capture at layer 16."""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments import config
from experiments.storage.schema import GenerationRecord


class ActivationCapture:
    """Captures layer-16 activations via HuggingFace output_hidden_states."""

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
    ) -> list[np.ndarray]:
        """Capture layer-16 activations for a list of generation records.

        For each record, constructs input_ids by concatenating prompt token IDs
        (from tokenizer) with stored gen_token_ids — never re-tokenizes decoded text.

        Returns list of numpy arrays, one per record, shape (num_gen_tokens, 4096).
        """
        all_activations: list[np.ndarray] = []

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            batch_activations = self._capture_single_batch(batch)
            all_activations.extend(batch_activations)

        return all_activations

    def _capture_single_batch(
        self,
        records: list[GenerationRecord],
    ) -> list[np.ndarray]:
        """Process a single batch of records through the model."""
        results: list[np.ndarray] = []

        # Process records individually to handle variable sequence lengths
        # (padding would require masking logic that complicates activation slicing)
        for record in records:
            # Reconstruct exact prompt tokens via chat template (prompt_text stores
            # user message content, not the chat-templated version)
            messages = [{"role": "user", "content": record.prompt_text}]
            prompt_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            prompt_length = len(prompt_ids)
            # Concatenate prompt token IDs with stored generation token IDs
            input_ids = prompt_ids + record.gen_token_ids
            input_tensor = torch.tensor(
                [input_ids],
                dtype=torch.long,
                device=self.model.device,
            )

            with torch.no_grad():
                outputs = self.model(input_tensor, output_hidden_states=True)

            # hidden_states[0] = embeddings, hidden_states[i+1] = layer i output
            # Layer 16 → index 17
            layer_hidden = outputs.hidden_states[config.HIDDEN_STATES_INDEX]

            # Slice to generated tokens only
            gen_activations = layer_hidden[:, prompt_length:, :]

            # Convert to float16 numpy, squeeze batch dim
            act_np = gen_activations.squeeze(0).cpu().numpy().astype(np.float16)
            results.append(act_np)

            # Fallback note: if memory spikes due to all 33 hidden states being
            # returned, switch to register_forward_hook on model.model.layers[16]
            # which captures only one layer's tensor.

        return results
