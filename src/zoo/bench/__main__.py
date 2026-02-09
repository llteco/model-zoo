#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse

from . import InputShape, benchmark
from .registry import BENCH

USAGE = """Benchmark modules.
uv -m zoo.bench {module_name} {input_shape1} {input_shape2} ... [options] [init args]

Example:
    uv -m zoo.bench conv2d 32,3,224,224 --kernel_size 3 --layers 10 --act "silu"
    uv -m zoo.bench sdpa 32,128,128 --num_heads 8
    uv -m zoo.bench gemm 1024,1024,1024
"""

parser = argparse.ArgumentParser(usage=USAGE)
parser.add_argument("module", nargs="?", choices=BENCH.list_all().keys())
parser.add_argument("input_shapes", nargs="*", action=InputShape)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--iters", type=int, default=10)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--reduce", choices=["mean", "median", "min", "max"])
parser.add_argument("--compile", action="store_true")
parser.add_argument("--half", action="store_true")
parser.add_argument(
    "--device", type=str, default="cuda", choices=["cpu", "cuda", "xpu"]
)
parser.add_argument("--cpu", nargs="?", dest="device", const="cpu")
parser.add_argument("--xpu", nargs="?", dest="device", const="xpu")
parser.add_argument("--cuda", nargs="?", dest="device", const="cuda")
parser.add_argument("--man", "-m", "-?", const="manual", nargs="?")


def main(argv=None) -> int:
    args, constructors = parser.parse_known_args(argv)
    if args.man == "manual":
        parser.print_help()
        return 0
    if args.man:
        BENCH.print(args.man)
        return 0

    times = benchmark(
        args.module,
        args.input_shapes,
        constructors,
        dynamo=args.compile,
        half=args.half,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        repeat=args.repeat,
        reduce=args.reduce,
    )
    print(args.reduce, times) if args.reduce else print(times)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
