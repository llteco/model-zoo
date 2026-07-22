"""
Copyright Wenyi Tang 2026

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import torch
from onnxifier.logger import info
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration
from transformers.cache_utils import DynamicCache, StaticCache
from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlCausalLMOutputWithPast

from .. import TRACE, GenerationConfig


@TRACE.register("lfm2.5vl")
class LFM25Trace(torch.nn.Module):
    def __init__(self, model_name: str = "LiquidAI/LFM2.5-VL-450M"):
        super().__init__()
        self.model = Lfm2VlForConditionalGeneration.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def generate(
        self,
        text: str,
        image: str | None = None,
        video: str | None = None,
    ):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            },
        ]
        if image:
            messages[0]["content"].append({"type": "image", "url": image})
        device = next(self.model.parameters()).device
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        resp = self.model.generate(**inputs, max_new_tokens=1024)
        return self.processor.decode(resp[0], skip_special_tokens=True)

    @torch.inference_mode()
    def forward(
        self,
        text: str,
        image: str | None = None,
        *,
        video: None = None,
        generation_config: GenerationConfig = GenerationConfig(),
    ):
        """Forward pass with tracing enabled."""

        assert video is None, "Video input is not supported for LFM2.5-VL"
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
        outputs: Lfm2VlCausalLMOutputWithPast = self.model(
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

    def process_outputs(self, outputs: Lfm2VlCausalLMOutputWithPast):
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
