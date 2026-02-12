#!/usr/bin/python
# -*- coding: UTF-8 -*-

import inspect
from typing import Any, Callable, Dict, Iterator, List, Optional, Type

from prettytable import PrettyTable

try:
    import torch.nn as nn
except ImportError:
    nn = None


class ParameterInfo:
    """存储参数的结构化信息"""

    def __init__(
        self,
        name: str,
        annotation: Optional[Type] = None,
        default: Any = inspect.Parameter.empty,
    ):
        self.name = name
        self.annotation = annotation
        self.default = default

    def __iter__(self) -> Iterator:
        """支持解包为 (name, annotation)，用于 argparse 等场景"""
        return iter((self.name, self.annotation, self.default))

    def __repr__(self) -> str:
        parts = [f"name={self.name}"]
        if self.annotation is not None:
            annotation_name = getattr(self.annotation, "__name__", str(self.annotation))
            parts.append(f"type={annotation_name}")
        if self.default != inspect.Parameter.empty:
            parts.append(f"default={self.default!r}")
        return f"ParameterInfo({', '.join(parts)})"

    def to_string(self) -> str:
        """转换为字符串表示，用于显示"""
        param_str = self.name

        # 添加类型注解
        if self.annotation is not None:
            annotation_name = getattr(self.annotation, "__name__", str(self.annotation))
            param_str += f": {annotation_name}"

        # 添加默认值
        if self.default != inspect.Parameter.empty:
            if isinstance(self.default, str):
                param_str += f"='{self.default}'"
            else:
                param_str += f"={self.default}"

        return param_str


class RegistryMetadata:
    """存储注册的类的元数据信息"""

    def __init__(
        self,
        name: str,
        cls: Type,
        init_signature: List[ParameterInfo],
        forward_signature: List[ParameterInfo],
    ):
        self.name = name
        self.cls = cls
        self.init_signature = init_signature
        self.forward_signature = forward_signature

    def __repr__(self) -> str:
        init_str = self._format_signature(self.init_signature)
        forward_str = self._format_signature(self.forward_signature)
        return (
            f"RegistryMetadata(name={self.name}, "
            f"init={init_str}, "
            f"forward={forward_str})"
        )

    @staticmethod
    def _format_signature(params: List[ParameterInfo]) -> str:
        """将参数列表格式化为字符串"""
        param_strs = [param.to_string() for param in params]
        return f"({', '.join(param_strs)})"

    def get_init_signature_str(self) -> str:
        """获取 __init__ 签名的字符串表示"""
        return self._format_signature(self.init_signature)

    def get_forward_signature_str(self) -> str:
        """获取 forward 签名的字符串表示"""
        return self._format_signature(self.forward_signature)


class Registry:
    """
    模型注册表，支持装饰器注册、元数据提取和表格形式输出。
    """

    def __init__(self, name: str = "registry"):
        """
        初始化Registry。

        Args:
            name: 注册表的名称
        """
        self.name = name
        self._registry: Dict[str, Type] = {}
        self._metadata: Dict[str, RegistryMetadata] = {}

    def register(self, module_name: Optional[str] = None) -> Callable:
        """
        装饰器，用于注册一个class到注册表。

        Args:
            module_name: 注册使用的名称，如果为None则使用class的__name__

        Returns:
            装饰器函数

        Raises:
            TypeError: 如果被注册的class不是nn.Module的子类
        """

        def decorator(cls: Type) -> Type:
            # 验证是nn.Module
            if nn is not None and not issubclass(cls, nn.Module):
                raise TypeError(
                    f"Class {cls.__name__} must be a subclass of torch.nn.Module, "
                    f"but got {cls.__bases__}"
                )

            # 使用指定的名称或class名称
            name = module_name if module_name is not None else cls.__name__

            # 检查是否已注册
            if name in self._registry:
                raise ValueError(
                    f"Class '{name}' is already registered in {self.name} registry"
                )

            # 注册类
            self._registry[name] = cls

            # 提取元数据
            self._metadata[name] = self._extract_metadata(name, cls)

            return cls

        return decorator

    @staticmethod
    def _extract_signature(method: Callable) -> List[ParameterInfo]:
        """
        提取方法的签名信息。

        Args:
            method: 要提取签名的方法

        Returns:
            参数信息列表
        """
        try:
            sig = inspect.signature(method)
            params = []
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue

                annotation = (
                    param.annotation
                    if param.annotation != inspect.Parameter.empty
                    else None
                )
                default = param.default

                params.append(
                    ParameterInfo(
                        name=param_name, annotation=annotation, default=default
                    )
                )

            return params
        except (ValueError, TypeError, AttributeError):
            # 如果提取失败，返回空列表
            return []

    @classmethod
    def _extract_metadata(cls, name: str, module_cls: Type) -> RegistryMetadata:
        """
        从类中提取元数据信息。

        Args:
            name: 注册名称
            module_cls: 要提取元数据的类

        Returns:
            RegistryMetadata对象
        """
        init_sig = cls._extract_signature(module_cls.__init__)
        forward_sig = cls._extract_signature(module_cls.forward)

        return RegistryMetadata(
            name=name,
            cls=module_cls,
            init_signature=init_sig,
            forward_signature=forward_sig,
        )

    def get(self, name: str) -> Type:
        """
        从注册表中获取已注册的类。

        Args:
            name: 注册名称

        Returns:
            注册的类

        Raises:
            KeyError: 如果类未被注册
        """
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def get_metadata(self, name: str) -> RegistryMetadata:
        """
        获取已注册类的元数据。

        Args:
            name: 注册名称

        Returns:
            RegistryMetadata对象
        """
        if name not in self._metadata:
            raise KeyError(f"'{name}' metadata not found in {self.name} registry")
        return self._metadata[name]

    def list_all(self) -> Dict[str, Type]:
        """
        列出所有已注册的类。

        Returns:
            包含所有注册项的字典
        """
        return self._registry.copy()

    def print(self, name: str) -> None:
        """
        打印指定模型的签名信息。

        Args:
            name: 注册名称

        Raises:
            KeyError: 如果模型未被注册
        """
        if name not in self._metadata:
            available = list(self._registry.keys())
            print(f"Error: '{name}' not found in {self.name} registry.")
            print(f"Available models: {', '.join(available)}")
            return

        metadata = self._metadata[name]
        print(f"\n{name}")
        print("=" * (len(name) + 4))
        print(f"\n__init__{metadata.get_init_signature_str()}")
        print(f"forward{metadata.get_forward_signature_str()}")
        print()

    def __contains__(self, name: str) -> bool:
        """检查名称是否已注册"""
        return name in self._registry

    def __len__(self) -> int:
        """获取已注册类的数量"""
        return len(self._registry)

    def __str__(self) -> str:
        """
        以表格形式输出注册表信息。

        Returns:
            格式化的表格字符串
        """
        if not self._registry:
            return f"Empty {self.name} Registry"
        table = PrettyTable()
        table.field_names = ["Name", "__init__ Signature", "forward Signature"]
        table.align["Name"] = "l"
        table.align["__init__ Signature"] = "l"
        table.align["forward Signature"] = "l"
        table.max_width["__init__ Signature"] = 40
        table.max_width["forward Signature"] = 40

        for name, metadata in self._metadata.items():
            table.add_row(
                [
                    name,
                    metadata.get_init_signature_str(),
                    metadata.get_forward_signature_str(),
                ]
            )

        return f"\n{self.name.upper()} REGISTRY:\n{table.get_string()}"

    def __repr__(self) -> str:
        registry_count = len(self._registry)
        return f"<{self.__class__.__name__} '{self.name}' with {registry_count} items>"
