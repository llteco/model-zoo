#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import contextlib
import importlib
import time
import types
from pathlib import Path
from typing import Literal, Optional, Union, get_args, get_origin

import torch
import torch.nn as nn

from .registry import BENCH


class InputShape(argparse.Action):
    """解析字符串形状描述并创建对应的 torch.Tensor。

    支持的格式：
        - "1,2,3" -> torch.empty([1,2,3])
        - "" -> torch.empty([])
        - "1024:float16" -> torch.empty([1024], dtype=torch.float16)
        - "2,3,4:float32" -> torch.empty([2,3,4], dtype=torch.float32)

    也支持 argparse 的 action 方式：
        parser.add_argument("--shape", type=str, action=InputShape)
    """

    # dtype 字符串到 torch.dtype 的映射
    DTYPE_MAP = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "float": torch.float32,
        "half": torch.float16,
        "double": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "int": torch.int32,
        "long": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
        "bfloat16": torch.bfloat16,
    }

    def __init__(self, *args, **kwargs):
        """初始化 InputShape。

        Args:
            shape_str: 形状描述字符串，如 "1,2,3" 或 "1024:float16"
        """
        if self._is_action_init(*args, **kwargs):
            super().__init__(*args, **kwargs)
            self._is_action = True
            self.shape_str = ""
            self.shape = []
            self.dtype = None
            return

        shape_str = self._get_shape_str(*args, **kwargs)
        self._is_action = False
        self.shape_str = shape_str
        self.shape = []
        self.dtype = None
        self._parse()

    @staticmethod
    def _is_action_init(*args, **kwargs) -> bool:
        if (
            len(args) >= 2
            and isinstance(args[0], (list, tuple))
            and isinstance(args[1], str)
        ):
            return True
        return "dest" in kwargs or "option_strings" in kwargs

    @staticmethod
    def _get_shape_str(*args, **kwargs) -> str:
        if len(args) == 1 and isinstance(args[0], str) and not kwargs:
            return args[0]
        if not args and "shape_str" in kwargs and isinstance(kwargs["shape_str"], str):
            return kwargs["shape_str"]
        raise TypeError(
            "InputShape expects a shape string or argparse Action parameters."
        )

    def __call__(
        self,
        _parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values,
        _option_string: Optional[str] = None,
    ) -> None:
        if isinstance(values, list):
            parsed = [InputShape(value) for value in values]
        elif isinstance(values, InputShape) and not values._is_action:
            parsed = values
        else:
            parsed = InputShape(values)
        setattr(namespace, self.dest, parsed)

    def _parse(self) -> None:
        """解析形状字符串。"""
        if not self.shape_str:
            # 空字符串，创建空 tensor
            self.shape = []
            return

        # 检查是否包含 dtype 说明（用冒号分隔）
        if ":" in self.shape_str:
            shape_part, dtype_part = self.shape_str.rsplit(":", 1)
            dtype_str = dtype_part.strip()
            if dtype_str in self.DTYPE_MAP:
                self.dtype = self.DTYPE_MAP[dtype_str]
            else:
                raise ValueError(
                    f"Unsupported dtype: {dtype_str}. "
                    f"Supported dtypes: {list(self.DTYPE_MAP.keys())}"
                )
        else:
            shape_part = self.shape_str

        # 解析形状维度
        if shape_part.strip():
            try:
                self.shape = [int(dim.strip()) for dim in shape_part.split(",")]
            except ValueError as e:
                raise ValueError(
                    f"Invalid shape string: {self.shape_str}. "
                    f"Expected comma-separated integers, got: {shape_part}"
                ) from e
        else:
            self.shape = []

    def to_tensor(self, device: Optional[str] = None) -> torch.Tensor:
        """创建对应的 torch.Tensor。

        Args:
            device: 设备类型，如 "cpu", "cuda" 等。如果为 None，使用默认设备。

        Returns:
            创建的 torch.Tensor
        """
        kwargs = {}
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype
        if device is not None:
            kwargs["device"] = device

        return torch.empty(self.shape, **kwargs)

    def __repr__(self) -> str:
        dtype_str = f", dtype={self.dtype}" if self.dtype else ""
        return f"InputShape(shape={self.shape}{dtype_str})"

    def __str__(self) -> str:
        return self.shape_str


def auto_import(module_name: str):
    """Import from `module_name` directory sources.

    Args:
        module_name (str): The name of the module to import from, e.g., "conv".
    """
    for f in Path(__file__).parent.glob(f"{module_name}/*.py"):
        importlib.import_module(f".{module_name}.{f.stem}", __package__)


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


def _get_argparse_config(annotation) -> dict:
    """从类型注解中提取 argparse 配置。

    支持的类型转换：
        - int, str, float, bool -> {"type": int/str/float/bool}
        - int | None / Optional[int] -> {"type": int}
        - tuple[int, ...] -> {"nargs": "+", "type": int}

    Args:
        annotation: 类型注解对象

    Returns:
        dict: 包含 argparse 参数的字典 (如 {"type": int} 或 {"nargs": "+", "type": int})
              如果无法识别类型，返回空字典
    """
    if annotation is None:
        return {}

    origin = get_origin(annotation)

    # 处理 Union 类型（包括 Optional 和 int | None）
    if origin is Union:
        args = get_args(annotation)
        # 过滤掉 NoneType
        non_none_types = [arg for arg in args if arg is not type(None)]
        if len(non_none_types) == 1:
            # 递归处理非 None 的类型
            return _get_argparse_config(non_none_types[0])
        return {}

    if isinstance(annotation, types.UnionType):
        args = get_args(annotation)
        non_none_types = [arg for arg in args if arg is not type(None)]
        if len(non_none_types) == 1:
            return _get_argparse_config(non_none_types[0])
        return {}

    # 处理 tuple 类型
    if origin is tuple:
        args = get_args(annotation)
        # tuple[int, ...] 表示可变长度
        if len(args) == 2 and args[1] is Ellipsis:
            element_type = args[0]
            if element_type in (int, str, float, bool):
                return {"nargs": "+", "type": element_type}
        return {}

    # 简单的内置类型
    if annotation in (int, str, float, bool):
        return {"type": annotation}

    return {}


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
        config = _get_argparse_config(value_type)
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
