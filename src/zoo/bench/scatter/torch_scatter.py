#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch

from ..registry import BENCH


@BENCH.register("torch_scatter")
class BenchTorchScatter(torch.nn.Module):
    """Benchmark for torch.scatter operation."""

    def __init__(
        self,
        dims: tuple[int, ...] = (1024, 1024, 1024),
        axis: int = 0,
        seed: int | None = None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.axis = axis
        # generate random index to scatter
        max_size = dims[axis]
        self.index = torch.randint(0, max_size, size=dims).long()
        # generate random src to scatter
        self.src = torch.rand(dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.index = self.index.to(x.device)
        self.src = self.src.to(x.device, dtype=x.dtype)
        return torch.scatter(x, self.axis, self.index, self.src)
