#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse

from . import TRACE, GenerationConfig, trace_model

USAGE = """Trace model inference.
uv run --extra=cuda -m zoo.trace {module_name} [options] [init args]

Example:
    uv run --extra=cuda -m zoo.trace qwen3.5 --text "Hello world"
    uv run --extra=cuda -m zoo.trace qwen3.5 --text "Hello" --image path/to/image.jpg
    uv run --extra=cuda -m zoo.trace qwen3.5 --mode decode --cache static
"""

parser = argparse.ArgumentParser(usage=USAGE)
parser.add_argument("module", nargs="?", choices=TRACE.list_all().keys())
parser.add_argument("--text", "-t", default="", help="Text prompt for the model")
parser.add_argument("--image", "-i", default=None, help="Path to input image")
parser.add_argument("--video", "-v", default=None, help="Path to input video")
parser.add_argument(
    "--mode",
    default="prefill",
    choices=["prefill", "decode"],
    help="Inference mode: prefill, decode",
)
parser.add_argument(
    "--seq-len", "-s", type=int, default=256, help="Sequence length for static mode"
)
parser.add_argument(
    "--cache",
    default="none",
    choices=["none", "static", "dynamic"],
    help="KV cache mode: none, static (fixed size), or dynamic",
)
parser.add_argument(
    "--capacity", type=int, default=1024, help="KV cache capacity for static mode"
)
parser.add_argument("--padding-side", choices=["left", "right"], help="Padding side")
parser.add_argument(
    "--device",
    default="cuda",
    choices=["cpu", "cuda"],
    help="Device to run on",
)
parser.add_argument(
    "--trace",
    "-x",
    nargs="*",
    help="specify modules to trace, e.g. model.model.decoder.layers.0.self_attn",
)
parser.add_argument(
    "--generate", action="store_true", help="use model.generate rather than forward"
)
parser.add_argument("--cpu", nargs="?", dest="device", const="cpu")
parser.add_argument("--cuda", nargs="?", dest="device", const="cuda")
parser.add_argument("--man", "-m", "-?", const="manual", nargs="?")


def main(argv=None) -> int:
    args, constructors = parser.parse_known_args(argv)
    if args.man == "manual":
        parser.print_help()
        return 0
    if args.man:
        TRACE.print(args.man)
        return 0

    generation_config = GenerationConfig(
        mode=args.mode,
        max_length=args.seq_len,
        cache=args.cache,
        capacity=args.capacity,
        padding_side=args.padding_side,
    )
    trace_model(
        args.module,
        text=args.text,
        image=args.image,
        video=args.video,
        config=generation_config,
        device=args.device,
        traces=args.trace,
        generate=args.generate,
        constructors=constructors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
