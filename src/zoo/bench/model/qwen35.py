"""
Copyright Wenyi Tang 2026

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import numpy as np
import torch
from transformers import AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5VisionModel,
)

from .. import BENCH


@BENCH.register("qwen3.5visual")
class Qwen3_5Visual(torch.nn.Module):
    def __init__(
        self,
        model_name="Qwen/Qwen3.5-0.8B",
        image_width: int = 256,
        image_height: int = 256,
    ):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3_5VisionModel.from_pretrained(
            model_name,
            key_mapping={"model.visual.": ""},
        )
        self.image_width = image_width
        self.image_height = image_height
        self.grid_thw = (1, 1, 1)

    @property
    def default_inputs(self):
        """Generate default inputs for benchmarking."""
        image = np.random.randint(
            0, 256, (self.image_height, self.image_width, 3), dtype=np.uint8
        )
        features = self.processor(images=image, text="", return_tensors="pt")
        self.grid_thw = features.image_grid_thw
        return dict(hidden_states=features.pixel_values)

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor):
        grid_thw = self.grid_thw
        return self.model(hidden_states, grid_thw).pooler_output


@BENCH.register("qwen3.5prefill")
class Qwen3_5Prefill(torch.nn.Module):
    def __init__(self, model_name="Qwen/Qwen3.5-0.8B"):
        super().__init__()
        self.qwen = Qwen3_5ForConditionalGeneration.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

    @property
    def default_inputs(self):
        """Generate default inputs for benchmarking."""
        text = "What is in this image?\n<|image_pad|>"
        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        features = self.processor(images=image, text=text, return_tensors="pt")
        return dict(**features)

    @torch.inference_mode()
    def forward(
        self,
        input_ids,
        attention_mask,
        mm_token_type_ids,
        pixel_values,
        image_grid_thw,
    ):
        return self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
        )


@BENCH.register("qwen3.5decode")
class Qwen3_5Decode(torch.nn.Module):
    def __init__(self, model_name="Qwen/Qwen3.5-0.8B"):
        super().__init__()
        self.qwen = Qwen3_5ForConditionalGeneration.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self._cached_kv = None
        self._cached_features = None

    @property
    def default_inputs(self):
        """Prepare prefilled KV cache for decode benchmark."""
        if self._cached_kv is not None:
            return self._prepare_decode_input()

        text = "What is in this image?\n<|image_pad|>"
        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        features = self.processor(images=image, text=text, return_tensors="pt")

        device = next(self.qwen.parameters()).device
        dtype = next(self.qwen.parameters()).dtype

        # Move features to device and cache them
        self._cached_features = {}
        for key, value in features.items():
            if not isinstance(value, torch.Tensor):
                self._cached_features[key] = value
                continue
            value = value.to(device=device)
            if key == "pixel_values" and torch.is_floating_point(value):
                value = value.to(dtype=dtype)
            self._cached_features[key] = value

        # Run prefill to get cached KV
        with torch.inference_mode():
            outputs = self.qwen(
                input_ids=self._cached_features["input_ids"],
                attention_mask=self._cached_features["attention_mask"],
                mm_token_type_ids=self._cached_features["mm_token_type_ids"],
                pixel_values=self._cached_features["pixel_values"],
                image_grid_thw=self._cached_features["image_grid_thw"],
                use_cache=True,
            )
            self._cached_kv = outputs.past_key_values

        return self._prepare_decode_input()

    def _prepare_decode_input(self):
        """Prepare a fresh decode input from cached KV."""
        device = next(self.qwen.parameters()).device

        # Return a single token that will be decoded
        return {
            "input_ids": torch.tensor(
                [[2]], dtype=torch.long, device=device
            ),  # token_id=2
            "attention_mask": torch.ones((1, 1), dtype=torch.long, device=device),
            "mm_token_type_ids": torch.zeros((1, 1), dtype=torch.long, device=device),
            "past_key_values": self._cached_kv,
        }

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask, mm_token_type_ids, past_key_values):
        # Decode: single token forward with pre-computed KV cache
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            past_key_values=past_key_values,
            use_cache=False,  # no need to update cache for benchmark
        )
        return outputs
