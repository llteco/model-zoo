#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..registry import BENCH


@triton.jit
def _vadd_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y
    tl.store(z_ptr + offsets, z, mask=mask)


@BENCH.register("triton_vadd")
class BenchTritonVAdd(nn.Module):
    """Benchmark for vector addition using Triton."""

    def __init__(self, block_size: int = 1024):
        super().__init__()
        self.block_size = block_size
        self.z = None

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.z is None:
            self.z = torch.empty_like(x + y)
        n_elements = x.numel()
        block_size = self.block_size
        grid_size = triton.cdiv(n_elements, block_size)
        _vadd_kernel[(grid_size,)](
            x,
            y,
            self.z,
            n_elements,
            BLOCK_SIZE=block_size,  # type: ignore
        )
        return self.z
