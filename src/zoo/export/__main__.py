#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse

from ..utils import InputShape
from . import export, EXPORT

USAGE = """Export modules.
uv -m zoo.export {module_name} {input_shape1} {input_shape2} ... [options] [init args]

Example:
    uv -m zoo.export bev.projection
"""

parser = argparse.ArgumentParser(usage=USAGE)
parser.add_argument("module", nargs="?", choices=EXPORT.list_all().keys())
parser.add_argument("input_shapes", nargs="*", action=InputShape)
parser.add_argument("--dynamo", action="store_true")
parser.add_argument("--opset-version", "-v", type=int, default=19)
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--cpu", nargs="?", dest="device", const="cpu")
parser.add_argument("--cuda", nargs="?", dest="device", const="cuda")
parser.add_argument("--external-directory", default=None)
parser.add_argument("--man", "-m", "-?", const="manual", nargs="?")


def main(argv=None) -> int:
    args, constructors = parser.parse_known_args(argv)
    if args.man == "manual":
        parser.print_help()
        return 0
    if args.man:
        EXPORT.print(args.man)
        return 0

    export(
        args.module,
        args.input_shapes,
        constructors,
        dynamo=args.dynamo,
        opset_version=args.opset_version,
        device=args.device,
        external_data=args.external_directory is not None,
        external_directory=args.external_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
