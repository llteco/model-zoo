#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse

import torch

from ..utils import InputShape
from . import EXPORT, export

USAGE = """Export modules.
uv -m zoo.export {module_name} {input_shape1} {input_shape2} ... [options] [init args]

Example:
    uv -m zoo.export bev.projection
"""

parser = argparse.ArgumentParser(usage=USAGE)
parser.add_argument("module", nargs="?", choices=EXPORT.available_names())
parser.add_argument("input_shapes", nargs="*", action=InputShape)
parser.add_argument("--dynamo", action="store_true")
parser.add_argument("--opset-version", "-v", type=int, default=23)
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--cpu", nargs="?", dest="device", const="cpu")
parser.add_argument("--cuda", nargs="?", dest="device", const="cuda")
parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"])
parser.add_argument("--float16", "-f16", nargs="?", dest="dtype", const="float16")
parser.add_argument("--bfloat16", "-bf16", nargs="?", dest="dtype", const="bfloat16")
parser.add_argument("--float32", "-f32", nargs="?", dest="dtype", const="float32")
parser.add_argument("--external-directory", "-d", default=None)
parser.add_argument(
    "--export-with-hier",
    "-hier",
    action="store_true",
    help="Export with hierarchical structure, exportable hier defined in <model>.hier",
)
parser.add_argument(
    "--no-post-process", action="store_true", help="Do not apply post process"
)
parser.add_argument("--man", "-m", "-?", const="manual", nargs="?")


def main(argv=None) -> int:
    args, constructors = parser.parse_known_args(argv)
    if args.man == "manual":
        parser.print_help()
        return 0
    if args.man:
        EXPORT.print(args.man)
        return 0
    dtype = None
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float32":
        dtype = torch.float32

    export(
        args.module,
        args.input_shapes,
        constructors,
        dynamo=args.dynamo,
        opset_version=args.opset_version,
        device=args.device,
        dtype=dtype,
        external_data=args.external_directory is not None,
        external_directory=args.external_directory,
        export_with_hier=args.export_with_hier,
        apply_post_process=not args.no_post_process,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
