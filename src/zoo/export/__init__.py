import argparse
import importlib
import os
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from hyperonnx import export_hyper_onnx

from ..registry import Registry
from ..utils import InputShape, get_argparse_config

EXPORT = Registry("EXPORT")


def auto_import(module_name: str):
    """Import from `module_name` directory sources.

    Args:
        module_name (str): The name of the module to import from, e.g., "conv".
    """
    for f in Path(__file__).parent.glob(f"{module_name}/**/*.py"):
        f = f.relative_to(Path(__file__).parent).with_suffix("").as_posix()
        f = f.replace("/", ".")
        importlib.import_module(f".{f}", __package__)


for module in ("vision", "vlm", "asr", "tts"):
    auto_import(module)


def create_module(module_name: str, constructors: list[str]) -> nn.Module:
    """Create a nn.Module instance based on the module name and constructor arguments.

    Args:
        module_name (str): The name of the module to create.
        constructors (list[str]): List of constructor arguments as strings.

    Returns:
        nn.Module: An instance of the requested module initialized with the provided
        arguments.
    """
    metadata = EXPORT.get_metadata(module_name)
    parser = argparse.ArgumentParser()
    for name, value_type, default_value in metadata.init_signature:
        config = get_argparse_config(value_type, default_value)
        parser.add_argument(f"--{name}", default=default_value, **config)
    args = parser.parse_args(constructors)
    # pylint: disable=protected-access
    return EXPORT.get(module_name)(**dict(args._get_kwargs()))


def export(
    module_name: str,
    input_shapes: list[InputShape],
    constructors: list[str],
    dynamo: bool = False,
    external_data: bool = False,
    external_directory: str | os.PathLike | None = None,
    opset_version: int = 19,
    device: Literal["cpu", "cuda"] = "cpu",
):
    """Export the module under test.

    Args:
        module_name (str): The name of the module to benchmark.
        input_shapes (list[InputShape]): List of input shapes to create inputs.
        constructors (list[str]): List of unparsed arguments for module __init__.
        dynamo (bool): Whether to use torch dynamo for the onnx export.
        hier (Sequence[str] | None): Specify a list of hierarchy to export.
        external_data (bool): Whether to use external data format for large models.
        external_directory (str | os.PathLike | None): Directory to store external data.
        opset_version (int): The ONNX opset version to use.
        device (Literal["cpu", "cuda"]): The device to run the export on.
    """
    model = create_module(module_name, constructors)
    model = model.to(device=device)
    if input_shapes:
        inputs = [shape.to_tensor(device=device) for shape in input_shapes]
    elif hasattr(model, "default_inputs"):
        inputs = list(getattr(model, "default_inputs", {}).values())
        inputs = [i.to(device=torch.device(device)) for i in inputs]
    else:
        inputs = []
    export_hyper_onnx(
        model,
        tuple(inputs),
        f"{module_name}.onnx",
        opset_version=opset_version,
        dynamo=dynamo,
        external_data=external_data,
        external_directory=external_directory,
        hiera=getattr(model, "hier", None),
        input_names=getattr(model, "input_names", None),
        output_names=getattr(model, "output_names", None),
    )
