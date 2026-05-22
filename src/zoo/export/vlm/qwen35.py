#!/usr/bin/python
# -*- coding: UTF-8 -*-
import fnmatch
import gc
from itertools import chain, product
from unittest.mock import patch

import torch
from hyperonnx.transformers import register_attention_opsets
from hyperonnx.transformers.attention import HYPERONNX_ATTN_IMPL
from hyperonnx.transformers.mamba import causal_conv1d_fn, register_mamba_opsets
from hyperonnx.transformers.recurrent import gated_delta_rule, register_recurrent_opsets
from onnxifier import OnnxGraph
from onnxifier.passes import PASSES
from onnxifier.passes.globals.reshape import reshape_model
from transformers import AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
    Qwen3_5VisionBlock,
    Qwen3_5VisionModel,
)

from .. import EXPORT, export_post_process

try:
    import modelopt.torch.opt as mto

    mto.enable_huggingface_checkpointing()
except ImportError:
    pass


@EXPORT.register("qwen3.5vision")
@export_post_process(
    [
        "fold_constant",
        "trt_vit_attention_replace",
        "eliminate_nop_slice",
        "eliminate_nop_concat",
        "eliminate_nop_pad",
        "eliminate_duplicated_transpose",
        "eliminate_dead_nodes",
        "remove_unused_functions",
    ]
)
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


FIXED_SHAPE_TEXT: dict = {
    "position_ids": [4, 1, 16],
    "attention_mask": [1, 16],
    "full_attention_mask": [1, 16, 1024],
    "inputs_embeds": [1, 16, 1024],
    "rope_rotary_cos_sin": [1, 16, 64],
    "context_lengths": [1],
    "kvcache_start_index": [1],
    "*key_value": [1, 2, 2, 1024, 256],
}


@PASSES.register("reshape_text_model")
def reshape_text_model(graph: OnnxGraph):
    shape_info = {}
    shape_wildcard = {}
    for k, v in FIXED_SHAPE_TEXT.items():
        if "*" in k or "?" in k:
            shape_wildcard[k] = v
        else:
            shape_info[k] = v
    for name, wildcard in product(chain(graph.inputs, graph.outputs), shape_wildcard):
        if fnmatch.fnmatch(name, wildcard):
            shape_info[name] = shape_wildcard[wildcard]
    return reshape_model(graph, shape_info=shape_info)


def _apply_mask_to_padding_states(hidden_states, attention_mask):
    """Mock apply_mask_to_padding_states without shape check."""
    if attention_mask is None:
        return hidden_states
    dtype = hidden_states.dtype
    return (hidden_states * attention_mask[:, :, None]).to(dtype)


def wrap_causal_conv1d(
    x,
    conv_state=None,
    weight=None,
    bias=None,
    activation: int = 0,
    seq_idx: int = 0,
):
    del activation, seq_idx
    assert weight is not None
    if weight.ndim == 2:
        weight.unsqueeze_(1)
    if bias is None:
        bias = torch.zeros([weight.shape[0]], dtype=weight.dtype, device=weight.device)
    if conv_state is None:
        conv_state = torch.zeros(
            [x.shape[0], weight.shape[0], weight.shape[-1]],
            dtype=x.dtype,
            device=x.device,
        )
    out, _ = causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_state,
        padding=weight.shape[-1] - 1,
        groups=weight.shape[0],
    )
    return out


@EXPORT.register("qwen3.5text")
@export_post_process
class Qwen3_5Text(torch.nn.Module):
    input_names = ("position_ids", "attention_mask", "inputs_embeds")
    output_names = ("logits",)
    fold_nodes_to_functions = False
    passes = [
        "infer_shape",
        "fold_constant",
        "attention_add_kvcache",
        "trt_attention_replace",
        "trt_causal_conv1d_replace",
        "trt_gated_delta_rule_replace",
        "fuse_int4_groupwise_gemm",
        "eliminate_nop_slice",
        "eliminate_nop_concat",
        "eliminate_nop_pad",
        "eliminate_duplicated_transpose",
        "eliminate_dead_nodes",
        "remove_unused_functions",
        "reshape_text_model",
    ]

    def __init__(
        self,
        model_name="Qwen/Qwen3.5-0.8B",
        seq_len: int = 256,
        capacity: int = 1024,
        use_lm_head: bool = False,
        use_full_attention_mask: bool = False,
        *,
        num_hidden_layers: int | None = None,  # debug purpose
    ):
        super().__init__()
        self.passes = Qwen3_5Text.passes.copy()
        self.seq_len = seq_len
        self.use_lm_head = use_lm_head
        if not use_lm_head:
            self.output_names = ("hidden_states",)
        if use_full_attention_mask:
            self.passes.insert(3, "attention_add_mask")
        # fix model shape
        FIXED_SHAPE_TEXT["position_ids"][2] = seq_len
        FIXED_SHAPE_TEXT["attention_mask"][1] = seq_len
        FIXED_SHAPE_TEXT["full_attention_mask"][1] = seq_len
        FIXED_SHAPE_TEXT["full_attention_mask"][2] = capacity
        FIXED_SHAPE_TEXT["inputs_embeds"][1] = seq_len
        FIXED_SHAPE_TEXT["rope_rotary_cos_sin"][1] = seq_len
        FIXED_SHAPE_TEXT["*key_value"][-2] = capacity
        register_attention_opsets()
        register_mamba_opsets()
        register_recurrent_opsets()
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name, attn_implementation=HYPERONNX_ATTN_IMPL
        )
        self.model = model.model.language_model

        def _replace_gdn_kernel(module):
            if isinstance(module, Qwen3_5GatedDeltaNet):
                module.causal_conv1d_fn = wrap_causal_conv1d
                module.causal_conv1d_update = wrap_causal_conv1d
                module.chunk_gated_delta_rule = gated_delta_rule
                module.recurrent_gated_delta_rule = gated_delta_rule

        self.model.apply(_replace_gdn_kernel)
        if num_hidden_layers is not None:
            self.model.config.num_hidden_layers = num_hidden_layers
        self.lm_head = model.lm_head
        self.patch = patch(
            "transformers.models.qwen3_5.modeling_qwen3_5.apply_mask_to_padding_states",
            _apply_mask_to_padding_states,
        )
        del model
        gc.collect()

    @property
    def default_inputs(self):
        seq_len = self.seq_len
        batch = 1
        position_ids = torch.arange(seq_len, dtype=torch.int64).view(1, 1, seq_len)
        position_ids = position_ids.expand(4, batch, -1)
        attention_mask = torch.ones([batch, seq_len], dtype=torch.bool)
        attention_mask[:, 0] = 0  # not all one
        inputs_embeds = torch.randn(
            [batch, seq_len, self.model.config.hidden_size], dtype=torch.bfloat16
        )
        return dict(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

    @torch.no_grad()
    def forward(self, position_ids, attention_mask, inputs_embeds):
        logits_to_keep = 1
        with self.patch:
            hidden_states = self.model(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=False,
            )[0]
        if self.use_lm_head:
            return self.lm_head(hidden_states[:, -logits_to_keep:])
        return hidden_states
