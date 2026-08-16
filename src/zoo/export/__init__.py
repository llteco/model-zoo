import argparse
import gc
import os
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import onnx
import torch
import torch.nn as nn
from hyperonnx import export_hyper_onnx
from onnxifier import OnnxGraph, PassManager
from onnxifier.utils import chdir

from ..registry import Registry
from ..utils import InputShape, get_argparse_config

EXPORT = Registry("EXPORT")

EXPORT.add_lazy_sources(Path(__file__).parent, __package__ or __name__)


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


def export_post_process(func_or_passes: list[str] | Callable):
    """Inject a default post_process helper into a class."""

    def _wrapper(cls):
        if callable(getattr(cls, "post_process", None)):
            return cls

        def post_process(
            self,
            onnx_file,
            *,
            opset_version=None,
            dynamo=None,
            external_data=None,
            external_directory=None,
            **kwargs,
        ):
            passes = getattr(
                cls,
                "passes",
                func_or_passes if isinstance(func_or_passes, Sequence) else [],
            )
            pm = PassManager(passes)
            graph = OnnxGraph(onnx.load_model(onnx_file, load_external_data=False))
            with TemporaryDirectory() as temp_dir:
                if not external_data:
                    graph.external_base = temp_dir
                with chdir(graph.external_base):
                    if not external_data:
                        graph.save_tensors_to_external(
                            Path(onnx_file).with_suffix(".data")
                        )
                    graph = pm.optimize(graph, recursive=True)
                    # restore in the final stage
                    if not external_data:
                        graph = PassManager(["restore_external_data"]).optimize(graph)
                graph.save(onnx_file)
                # hyperonnx dumps <Type>_combined.onnx before functions are
                # composed into the graph; overwrite it with the post-processed
                # model so the combined artifact keeps its FunctionProtos.
                # Inline external data: downstream tools may copy the bare
                # .onnx without its sidecar.
                if external_directory:
                    for combined in Path(external_directory).glob("*_combined.onnx"):
                        onnx.save_model(
                            onnx.load_model(onnx_file),
                            combined,
                            save_as_external_data=False,
                        )

        setattr(cls, "post_process", post_process)
        return cls

    if isinstance(func_or_passes, Sequence):
        return _wrapper
    return _wrapper(func_or_passes)


def i2i_f2f(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast tensor to dtype without casting integer to float or vice versa."""

    with suppress(ValueError, TypeError):
        torch.finfo(t.dtype)
        torch.finfo(dtype)
        return t.to(dtype)
    with suppress(ValueError, TypeError):
        torch.iinfo(t.dtype)
        torch.iinfo(dtype)
        return t.to(dtype)
    return t


def export(
    module_name: str,
    input_shapes: list[InputShape],
    constructors: list[str],
    dynamo: bool = False,
    external_data: bool = False,
    external_directory: str | os.PathLike | None = None,
    opset_version: int = 19,
    device: Literal["cpu", "cuda"] = "cpu",
    dtype: torch.dtype | None = None,
    export_with_hier: bool = False,
    apply_post_process: bool = True,
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
        dtype (torch.dtype | None): The data type to use for the model parameters and
            inputs. If None, the default data type of the model parameters will be used.
        export_with_hier (bool): Whether to export with hierarchical structure,
            exportable hier defined in <model>.hier.
        apply_post_process (bool): Whether to apply post process after export.
    """
    model = create_module(module_name, constructors)
    model = model.to(device=device)
    if dtype:
        model = model.to(dtype=dtype)
    if input_shapes:
        inputs = [shape.to_tensor(device=device) for shape in input_shapes]
    elif hasattr(model, "default_inputs"):
        inputs = list(getattr(model, "default_inputs", {}).values())
        inputs = [i.to(device=device) if hasattr(i, "to") else i for i in inputs]
        if dtype:
            inputs = [i2i_f2f(i, dtype) if hasattr(i, "to") else i for i in inputs]
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
        hiera=getattr(model, "hier", None) if export_with_hier else None,
        compile_hier=getattr(model, "compile_hier", None) if export_with_hier else None,
        input_names=getattr(model, "input_names", None),
        output_names=getattr(model, "output_names", None),
        fold_nodes_to_functions=getattr(model, "fold_nodes_to_functions", True),
        cutlass_tune=getattr(model, "cutlass_tune", True),
    )
    if apply_post_process and callable(getattr(model, "post_process", None)):
        fn = getattr(model, "post_process")
        del model
        gc.collect()
        fn(
            f"{module_name}.onnx",
            opset_version=opset_version,
            dynamo=dynamo,
            external_data=external_data,
            external_directory=external_directory,
        )
