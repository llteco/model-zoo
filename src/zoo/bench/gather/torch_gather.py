#!/usr/bin/python
# -*- coding: UTF-8 -*-

from typing import Literal, Optional

import torch

from ..registry import BENCH


@BENCH.register("torch_gather")
class BenchTorchGather(torch.nn.Module):
    """Benchmark for torch.gather operation."""

    def __init__(
        self,
        dims: tuple[int, ...] = (1024, 1024, 1024),
        axis: int = 0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.axis = axis
        # generate random index to gather
        max_size = dims[axis]
        self.index = torch.randint(0, max_size, size=dims).long()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.index = self.index.to(x.device)
        return torch.gather(x, self.axis, self.index)


@BENCH.register("grid_sample")
class BenchGridSample(torch.nn.Module):
    """Benchmark for torch.nn.functional.grid_sample operation."""

    def __init__(
        self,
        dims: tuple[int, ...] = (1, 32, 512, 512),
        mode: Literal["bilinear", "nearest", "bicubic"] = "bilinear",
        padding_mode: Literal["zeros", "border", "reflection"] = "zeros",
        align_corners: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        # generate random grid to sample
        spatial = len(dims) - 2
        self.grid = torch.rand(dims[0], dims[2], dims[3], spatial) * 2 - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.grid = self.grid.to(x.device)
        return torch.nn.functional.grid_sample(
            x,
            self.grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
