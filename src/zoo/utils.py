#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import types
from typing import Any, Optional, Union, get_args, get_origin

import torch


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


def get_argparse_config(annotation: Any, default_value: Any) -> dict:
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
            return get_argparse_config(non_none_types[0], default_value)
        return {}

    if isinstance(annotation, types.UnionType):
        args = get_args(annotation)
        non_none_types = [arg for arg in args if arg is not type(None)]
        if len(non_none_types) == 1:
            return get_argparse_config(non_none_types[0], default_value)
        return {}

    # 处理 tuple 类型
    if origin is tuple:
        args = get_args(annotation)
        # tuple[int, ...] 表示可变长度
        if len(args) == 2 and args[1] is Ellipsis:
            element_type = args[0]
            if element_type in (int, str, float, bool):
                return {"nargs": "+", "type": element_type}
        else:
            element_type = args[0]
            if element_type in (int, str, float, bool):
                return {"nargs": len(args), "type": element_type}
        return {}

    # 简单的内置类型
    if annotation in (int, str, float):
        return {"type": annotation}
    if annotation is bool:
        if default_value:
            return {"action": "store_false"}
        else:
            return {"action": "store_true"}
    return {}
