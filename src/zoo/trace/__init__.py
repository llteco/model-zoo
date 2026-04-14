import sys
from collections import defaultdict
from collections.abc import Sequence
from functools import wraps


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
                print(f"({name}) #{i}: {a.shape}, {a.dtype}")
            else:
                print(f"({name}) #{i}: {a}")
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                print(f"({name}) {k}: {v.shape}, {v.dtype}")
            else:
                print(f"({name}) {k}: {v}")
        fn = getattr(obj, f"__{method.__name__}__", None)
        if fn is not None:
            return fn(*args, **kwargs)
        raise RuntimeError(f"Method {method.__name__} not found in {obj}")

    if obj is not None:
        setattr(obj, f"__{method.__name__}__", method)
        setattr(obj, method.__name__, _wrapper)

    return _wrapper
