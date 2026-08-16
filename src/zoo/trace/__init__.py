import argparse
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Literal

import torch
from onnxifier.logger import debug, error, info, trace, warning

from ..registry import Registry
from ..utils import get_argparse_config

TRACE = Registry("TRACE")

TRACE.add_lazy_sources(Path(__file__).parent, __package__ or __name__)


def create_module(module_name: str, constructors: list[str]):
    """Create a nn.Module instance based on the module name and constructor arguments.

    Args:
        module_name (str): The name of the module to create.
        constructors (list[str]): List of constructor arguments as strings.

    Returns:
        nn.Module: An instance of the requested module initialized with the provided
        arguments.
    """
    metadata = TRACE.get_metadata(module_name)
    parser = argparse.ArgumentParser()
    for name, value_type, default_value in metadata.init_signature:
        config = get_argparse_config(value_type, default_value)
        parser.add_argument(f"--{name}", default=default_value, **config)
    args = parser.parse_args(constructors)
    # pylint: disable=protected-access
    return TRACE.get(module_name)(**dict(args._get_kwargs()))


@dataclass
class GenerationConfig:
    mode: Literal["prefill", "decode"] = "prefill"
    max_length: int = 256
    cache: None | Literal["static", "dynamic"] = None
    capacity: int = 1024  # for static cache
    padding_side: Literal["left", "right"] = "right"


def return_with_locals(fn, vars: Sequence[str], *, return_as_tuple: bool = True):
    """Wrap a function so its return value becomes (original_output, {local vars}).

    Args:
        fn: The function to wrap.
        vars: List of local variable names to capture from the function's scope.

    Returns:
        A wrapped function that returns (original_output, captured_locals_dict).
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        captured: dict = defaultdict(lambda: None, {v: None for v in vars})
        old_trace = sys.gettrace()

        def _trace(frame, event, _arg):
            if event == "return":
                for v in vars:
                    if v in frame.f_locals:
                        captured[v] = frame.f_locals[v]
            return _trace

        try:
            sys.settrace(_trace)
            result = fn(*args, **kwargs)
        finally:
            sys.settrace(old_trace)

        if return_as_tuple:
            return result, *tuple(captured.values())
        return result, captured

    return wrapper


def trace_method(method, *, obj=None):
    obj = obj or getattr(method, "__self__", None)
    assert obj is not None
    name = obj.__class__.__name__

    @wraps(method)
    def _wrapper(*args, **kwargs):
        import torch  # lazy import

        for i, a in enumerate(args):
            if isinstance(a, torch.Tensor):
                info(f"({name}) #{i}: {a.shape}, {a.dtype}")
                debug(
                    "(%s) #%d: min=%s max=%s nan=%d",
                    name,
                    i,
                    a.min(),
                    a.max(),
                    a.isnan().sum(),
                )
            else:
                info(f"({name}) #{i}: {a}")
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                info(f"({name}) {k}: {v.shape}, {v.dtype}")
                debug(
                    "(%s) %s: min=%s max=%s nan=%d",
                    name,
                    k,
                    v.min(),
                    v.max(),
                    v.isnan().sum(),
                )
            else:
                info(f"({name}) {k}: {v}")
        fn = getattr(obj, f"__{method.__name__}__", None)
        if fn is not None:
            return fn(*args, **kwargs)
        raise RuntimeError(f"Method {method.__name__} not found in {obj}")

    if obj is not None:
        setattr(obj, f"__{method.__name__}__", method)
        setattr(obj, method.__name__, _wrapper)

    return _wrapper


def trace_model(
    module_name: str,
    text: str = "",
    image: str | None = None,
    video: str | None = None,
    config: GenerationConfig = GenerationConfig(),
    device: str = "cuda",
    traces: list[str] | None = None,
    generate: bool = False,
    constructors: list[str] | None = None,
):
    """Run trace on a registered model.

    Args:
        module_name: Name of the registered module
        text: Text prompt
        image: Path to image file
        video: Path to video file
        config: Generation configuration
        device: Device to run on (cpu, cuda)
        traces: List of module names to trace
        generate: Whether to use model.generate instead of forward
        constructors: Additional constructor arguments from CLI
    """

    if module_name not in TRACE:
        error(f"Module '{module_name}' not found in TRACE registry.")
        error(f"Available modules: {TRACE.available_names()}")
        return

    # Instantiate model
    model = create_module(module_name, constructors or [])
    model.to(device)

    # Apply trace_method to model forward
    trace_method(model.forward, obj=model)
    for name, module in model.named_modules():
        # remove 'model.' prefix if exists
        striped_name = re.sub(r"^model\.?", "", name)
        if traces and (name in traces or striped_name in traces):
            trace_method(module.forward, obj=module)
        if name:
            trace(name)

    # Run forward with traced inputs
    with torch.inference_mode():
        if generate:
            outputs = model.generate(
                text=text,
                image=image,
                video=video,
            )
            info("%s", outputs)
            return
        else:
            outputs = model.forward(
                text=text,
                image=image,
                video=video,
                generation_config=config,
            )

    # Print output for verification
    if callable(getattr(model, "process_outputs", None)):
        getattr(model, "process_outputs")(outputs)
    else:
        warning("%s doesn't implement process_outputs method")
        info("Raw outputs: [%s]", outputs)
