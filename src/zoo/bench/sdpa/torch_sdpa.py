#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch

from ..registry import BENCH


@BENCH.register("torch_sdpa")
class BenchTorchSDPA(torch.nn.Module):
    """Benchmark for torch.nn.functional.scaled_dot_product_attention"""

    def __init__(self, is_causal: bool = True, enable_gqa: bool = False):
        super().__init__()
        self.is_causal = is_causal
        self.enable_gqa = enable_gqa

    def forward(self, hidden_states, attn_mask=None):
        # pylint: disable=not-callable
        return torch.nn.functional.scaled_dot_product_attention(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=attn_mask,
            is_causal=self.is_causal,
            enable_gqa=self.enable_gqa,
        )
