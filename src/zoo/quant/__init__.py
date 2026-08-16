import argparse
import os
from pathlib import Path
from typing import Literal

from onnxifier.logger import set_level

from ..registry import Registry
from ..utils import get_argparse_config

QUANT = Registry("QUANT")

QUANT.add_lazy_sources(Path(__file__).parent, __package__ or __name__)


def create_module(module_name: str, constructors: list[str]):
    """Create a nn.Module instance based on the module name and constructor arguments.

    Args:
        module_name (str): The name of the module to create.
        constructors (list[str]): List of constructor arguments as strings.

    Returns:
        nn.Module: An instance of the requested module initialized with the provided
        arguments.
    """
    metadata = QUANT.get_metadata(module_name)
    parser = argparse.ArgumentParser()
    for name, value_type, default_value in metadata.init_signature:
        config = get_argparse_config(value_type, default_value)
        parser.add_argument(f"--{name}", default=default_value, **config)
    args = parser.parse_args(constructors)
    # pylint: disable=protected-access
    return QUANT.get(module_name)(**dict(args._get_kwargs()))


def quant(
    model_id: str | os.PathLike,
    recipe: str | os.PathLike,
    constructors: list[str] | None = None,
    output_dir: str | os.PathLike = ".",
    verbose: bool = False,
    device: Literal["cpu", "cuda"] = "cpu",
):
    """Quantize the module under test.

    Args:
        model_id (str | os.PathLike): The identifier or path of the model to quantize.
        quant_framework (str): Quantization framework to use.
        recipe (str | os.PathLike): Path to the quantization recipe config file.
        constructors (list[str] | None): List of unparsed arguments for module __init__.
        output_dir (str | os.PathLike): Directory to save quantized models.
        verbose (bool): Whether to enable verbose logging.
        device (Literal["cpu", "cuda"]): The device to run quantization on.
    """
    if verbose:
        set_level("debug")
    model = create_module(str(model_id), constructors or [])
    model = model.to(device=device)
    model.quant(output_dir)
