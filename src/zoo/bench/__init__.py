#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import contextlib
import importlib
import time
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from ..utils import InputShape, get_argparse_config
from .registry import BENCH


def auto_import(module_name: str):
    """Import from `module_name` directory sources.

    Args:
        module_name (str): The name of the module to import from, e.g., "conv".
    """
    for f in Path(__file__).parent.glob(f"{module_name}/**/*.py"):
        f = f.relative_to(Path(__file__).parent).with_suffix("").as_posix()
        f = f.replace("/", ".")
        importlib.import_module(f".{f}", __package__)


for module in ("conv", "sdpa", "gemm", "mlp", "gather", "scatter", "vadd", "model"):
    auto_import(module)


@contextlib.contextmanager
def timer(device: torch.device, iters: int, times: list[float]):
    """Context manager for timing code execution.

    Args:
        device (torch.device): The device on which the code runs.
        iters (int): Number of iterations to average the timing over.
        times (list[float]): List to append the measured times (in milliseconds).

    Yields:
        None: The context manager does not yield any value.
    """
    if device.type == "cpu":
        start = time.perf_counter_ns()
        yield
        end = time.perf_counter_ns()
        times.append((end - start) / iters / 1e6)
    else:
        start = torch.Event(device=device, enable_timing=True)
        end = torch.Event(device=device, enable_timing=True)
        start.record()
        yield
        end.record()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "xpu":
            torch.xpu.synchronize()
        times.append(start.elapsed_time(end) / iters)


def create_module(module_name: str, constructors: list[str]) -> nn.Module:
    """Create a nn.Module instance based on the module name and constructor arguments.

    Args:
        module_name (str): The name of the module to create.
        constructors (list[str]): List of constructor arguments as strings.

    Returns:
        nn.Module: An instance of the requested module initialized with the provided
        arguments.
    """
    metadata = BENCH.get_metadata(module_name)
    parser = argparse.ArgumentParser()
    for name, value_type, default_value in metadata.init_signature:
        config = get_argparse_config(value_type, default_value)
        parser.add_argument(f"--{name}", default=default_value, **config)
    args = parser.parse_args(constructors)
    # pylint: disable=protected-access
    return BENCH.get(module_name)(**dict(args._get_kwargs()))


def benchmark(
    module_name: str,
    input_shapes: list[InputShape],
    constructors: list[str],
    dynamo: bool = False,
    half: bool = False,
    device: str = "cuda",
    warmup: int = 5,
    iters: int = 10,
    repeat: int = 1,
    reduce: Literal["mean", "median", "min", "max"] | None = None,
):
    """Benchmark the module under test.

    Args:
        module_name (str): The name of the module to benchmark.
        input_shapes (list[InputShape]): List of input shapes to create inputs.
        constructors (list[str]): List of unparsed arguments for module __init__.
        dynamo (bool): Whether to use torch.compile for the module.
        half (bool): Whether to convert the module to half precision.
        device (str): The device to run the benchmark on ("cpu", "cuda", "xpu").
        warmup (int): Number of warmup iterations to run before timing.
        iters (int): Number of iterations to run for timing.
        repeat (int): Number of times to repeat the timing for averaging.
        reduce (str | None): Method to reduce the timing results ("mean", "median",
            "min", "max") or None to return all times. Defaults to None.
    """
    model = create_module(module_name, constructors)
    model = model.to(device=device)
    if half:
        model = model.half()
    if dynamo:
        model = torch.compile(model)
    with torch.inference_mode():
        if input_shapes:
            inputs = [shape.to_tensor(device=device) for shape in input_shapes]
        elif hasattr(model, "default_inputs"):
            inputs = list(getattr(model, "default_inputs", {}).values())
            inputs = [i.to(device=torch.device(device)) for i in inputs]
            if half:
                inputs = [i.half() for i in inputs]
        else:
            inputs = []
        for _ in range(warmup):
            model(*inputs)
        times = []
        try:
            for _ in range(repeat):
                with timer(torch.device(device), iters, times):
                    for _ in range(iters):
                        model(*inputs)
        except KeyboardInterrupt:
            pass
    if not times:
        return None if reduce else times
    if reduce == "mean":
        return sum(times) / len(times)
    if reduce == "median":
        sorted_times = sorted(times)
        n = len(sorted_times)
        if n % 2 == 1:
            return sorted_times[n // 2]
        return (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
    if reduce == "min":
        return min(times)
    if reduce == "max":
        return max(times)
    return times
