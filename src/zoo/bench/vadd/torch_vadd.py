#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
from torch.nn import Module

from ..registry import BENCH


@BENCH.register("torch_vadd")
class BenchTorchVAdd(Module):
    """Benchmark for vector addition using PyTorch."""

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return torch.add(x, y)
