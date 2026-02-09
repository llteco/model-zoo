#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch.nn as nn


def get_act(act_name: str):
    """Get activation function by name."""
    act_name = act_name.lower()
    if act_name == "relu":
        return nn.ReLU(inplace=True)
    elif act_name == "gelu":
        return nn.GELU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "elu":
        return nn.ELU()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation function: {act_name}")
