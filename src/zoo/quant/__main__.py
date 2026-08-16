#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse

from . import QUANT, quant

USAGE = """Quantize modules.
uv -m zoo.quant {module_name} -qf {framework} -r {recipe} [options] [init args]

Example:
    uv -m zoo.quant qwen3_vl
"""

parser = argparse.ArgumentParser(usage=USAGE)
parser.add_argument("module", nargs="?", choices=QUANT.available_names())
parser.add_argument(
    "--recipe",
    "-r",
    help="Path to the quantization recipe config file",
)
parser.add_argument(
    "-o",
    "--output-dir",
    default="./quantized_models",
    help="Directory to save quantized models",
)
parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="Enable verbose logging",
)
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
parser.add_argument("--cpu", nargs="?", dest="device", const="cpu")
parser.add_argument("--cuda", nargs="?", dest="device", const="cuda")
parser.add_argument("--man", "-m", "-?", const="manual", nargs="?")


def main(argv=None) -> int:
    args, constructors = parser.parse_known_args(argv)
    if args.man == "manual":
        parser.print_help()
        return 0
    if args.man:
        QUANT.print(args.man)
        return 0

    quant(
        args.module,
        recipe=args.recipe,
        constructors=constructors,
        output_dir=args.output_dir,
        verbose=args.verbose,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
