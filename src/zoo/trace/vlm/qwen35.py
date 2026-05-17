"""
Copyright Wenyi Tang 2026

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from unittest.mock import patch

import torch
from onnxifier.logger import info
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.cache_utils import DynamicCache, StaticCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5CausalLMOutputWithPast

from .. import TRACE, GenerationConfig


def _apply_mask_to_padding_states(hidden_states, attention_mask):
    """Mock apply_mask_to_padding_states without shape check."""
    if attention_mask is None:
        return hidden_states
    dtype = hidden_states.dtype
    return (hidden_states * attention_mask[:, :, None]).to(dtype)


@TRACE.register("qwen3.5")
class Qwen3_5Trace(torch.nn.Module):
    """Qwen3.5 trace implementation for VLM inference.

    Supports multiple inference modes:
    - prefill: Full prefill with optional image
    - decode: Single token decode with KV cache
    - static_seqlen: Prefill with fixed sequence length (left padding)
    """

    def __init__(self, model_name: str = "Qwen/Qwen3.5-0.8B"):
        super().__init__()
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def generate(
        self,
        text: str,
        image: str | None = None,
        video: str | None = None,
    ):
        inputs = self.processor(
            text=text, images=image, videos=video, return_tensors="pt"
        )
        inputs = inputs.to(next(self.model.parameters()).device)
        resp = self.model.generate(**inputs, max_new_tokens=1024)
        return self.processor.decode(resp[0], skip_special_tokens=True)

    def forward(
        self,
        text: str,
        image: str | None = None,
        video: str | None = None,
        *,
        generation_config: GenerationConfig = GenerationConfig(),
    ):
        """Forward pass with tracing enabled."""
        if generation_config.cache == "dynamic":
            info("Use dynamic cache")
            cache = DynamicCache()
        elif generation_config.cache == "static":
            info("Use static cache(cap=%d)", generation_config.capacity)
            cache = StaticCache(
                self.model.config, max_cache_len=generation_config.capacity
            )
        else:
            info("Disable KV cache")
            cache = None
        padding = False
        if generation_config.padding_side:
            info("Use %s padding", generation_config.padding_side)
            self.processor.tokenizer.padding_side = generation_config.padding_side
            padding = "max_length"
        device = next(self.model.parameters()).device
        inputs = self.processor(
            text=text,
            images=image,
            videos=video,
            return_tensors="pt",
            padding=padding,
            max_length=generation_config.max_length,
        )
        inputs = inputs.to(device)
        # need to give correct position ids when use padding
        if generation_config.padding_side:
            position_ids = torch.arange(inputs.input_ids.shape[1], device=device)
            position_ids = position_ids.unsqueeze(0).expand_as(inputs.input_ids)
            position_ids.masked_fill_(~inputs.attention_mask.bool(), 0)
            if generation_config.padding_side == "left":
                position_ids = position_ids - position_ids.nonzero().min(0).values[-1]
                position_ids.masked_fill_(~inputs.attention_mask.bool(), 0)
            inputs["position_ids"] = position_ids

        # apply masking to linear attention states
        with patch(
            "transformers.models.qwen3_5.modeling_qwen3_5.apply_mask_to_padding_states",
            _apply_mask_to_padding_states,
        ):
            outputs: Qwen3_5CausalLMOutputWithPast = self.model(
                **inputs, past_key_values=cache, use_cache=cache is not None
            )
        # extract last logits
        logits = outputs.logits
        assert logits is not None
        valid_tokens = inputs.attention_mask.sum(1).squeeze()
        if generation_config.padding_side == "right":
            outputs["logits"] = logits[:, :valid_tokens, :]
        else:
            outputs["logits"] = logits[:, -valid_tokens:, :]
        return outputs

    def process_outputs(self, outputs: Qwen3_5CausalLMOutputWithPast):
        logits = outputs.logits
        last_token_logits = logits[:, -1, :]
        # Get top 5 tokens
        top_k = 5
        top_probs = torch.softmax(last_token_logits.float(), dim=-1)
        top_tokens = torch.topk(top_probs, top_k)
        print("\n=== Output ===")
        print(f"Logits shape: {logits.shape}")
        print(f"Top {top_k} predicted tokens:")
        # Try to decode tokens if processor/tokenizer is available
        top_indices = top_tokens.indices[0]
        top_values = top_tokens.values[0]
        tokenizer = getattr(self.processor, "tokenizer")
        for i, (prob, token_id) in enumerate(zip(top_values, top_indices)):
            token_text = tokenizer.decode([token_id.item()])
            # Escape special characters for display
            token_repr = token_text.replace("\n", "\\n")
            token_repr = token_repr.replace("\t", "\\t").replace(" ", "␣")
            tid = token_id.item()
            print(f"  {i + 1}. id={tid}, prob={prob.item():.4f}, '{token_repr}'")
