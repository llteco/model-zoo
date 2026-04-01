#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
from hyperonnx.transformers import register_attention_opsets
from hyperonnx.transformers.attention import HYPERONNX_ATTN_IMPL
from transformers import AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5VisionBlock,
    Qwen3_5VisionModel,
)

from .. import EXPORT


@EXPORT.register("qwen3.5vision")
class Qwen3_5Vision(torch.nn.Module):
    hier = [Qwen3_5VisionBlock]
    input_names = ("image_features",)
    output_names = ("pooler_output",)

    def __init__(
        self,
        model_name="Qwen/Qwen3.5-0.8B",
        image_width: int = 256,
        image_height: int = 256,
    ):
        super().__init__()
        register_attention_opsets()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3_5VisionModel.from_pretrained(
            model_name,
            attn_implementation=HYPERONNX_ATTN_IMPL,
            key_mapping={"model.visual.": ""},
        )
        self.image_width = image_width
        self.image_height = image_height
        self.grid_thw = (1, 1, 1)

    @property
    def default_inputs(self):
        """Generate default inputs for benchmarking."""
        image = torch.randint(
            0, 256, (self.image_height, self.image_width, 3), dtype=torch.uint8
        )
        features = self.processor(images=image, text="", return_tensors="pt")
        self.grid_thw = features.image_grid_thw
        return dict(hidden_states=features.pixel_values)

    def forward(self, hidden_states: torch.Tensor):
        grid_thw = self.grid_thw
        return self.model(hidden_states, grid_thw).pooler_output
