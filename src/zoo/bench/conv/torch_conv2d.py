#!/usr/bin/python
# -*- coding: UTF-8 -*-

from typing import Literal

import torch.nn as nn

from ..registry import BENCH
from .act import get_act


@BENCH.register("conv2d")
class BenchConv2d(nn.Module):
    """Benchmark for 2D convolution layers.

    Args:
        channels (int): Number of input and output channels. Default is 256.
        layers (int): Number of convolutional layers. Default is 3.
        kernel_size (int): Size of the convolutional kernel. Default is 3.
        act (str | None): Activation function to use after each convolutional layer.
            Supported values are "relu", "gelu", "silu", "elu", "sigmoid", "tanh".
            If None, no activation is applied. Default is None.
    """

    def __init__(
        self,
        channels: int = 256,
        layers: int = 3,
        kernel_size: int = 3,
        act: Literal["relu", "gelu", "silu", "elu", "sigmoid", "tanh"] | None = None,
    ):
        super(BenchConv2d, self).__init__()
        nets = []
        for _ in range(layers):
            nets.append(
                nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2),
            )
            if act is not None:
                nets.append(get_act(act))
        self.nets = nn.Sequential(*nets)

    def forward(self, x):
        return self.nets(x)
