"""Build tooling for the ReSplat Triton-plugin engine variant.

The stage0 graph's cost-volume segment (warp sampling + correlation,
GridSample..ReduceMean) is TensorRT's peak-memory node: myelin fuses it into
one segment with ~2.4 GiB of tactic scratch (the stage0 memory peak on a
12 GB card; graph-level chunking cannot split it — TRT re-merges sibling
branches). This module replaces that segment with ONE ``trt::triton``
plugin node executing a hand-written online-reduction Triton kernel
(zoo.model.resplat.unimatch.cost_volume_triton): zero intermediates, so the
plugin's workspace is ~0 and the engine keeps the no-cost-volume scratch
level (1.49 GiB) in a SINGLE graph — no A/B split, no duplicate IO.

Pipeline (``python -m zoo.trace.resplat_build triton <engines_dir>``):

1. compile the kernel to a cubin (``warmup`` on misaligned views so the
   cubin carries no 16B-alignment assumption) and hand-write the v2 kernel
   bundle manifest (compile_hier cannot capture hand-written @triton.jit
   kernels, and inductor-generated bundles carry per-chunk intermediates
   that the plugin's no-reuse workspace sizing would sum to ~8 GiB);
2. patch the original stage0 ONNX: drop the 18-node cost-volume segment,
   insert the ``trt::triton`` node producing the original ReduceMean
   output name (downstream wiring unchanged; the camera-grid math stays
   in the graph — it was never the memory problem);
3. build the fp16 engine (nv dump) and stage it as ``stage0.trtbin``
   next to the step engine. The bundle is serialized INTO the engine —
   the on-disk bundle is only needed at build time.

Numerics: the kernel is bit-exact vs the fp32 warp+correlation reference
(grid_sample_manual), including fp16-Inf grid coordinates from
behind-camera rays (saturated in-kernel). Parity vs the stock fp16 engine
matches the split-variant class (the replaced segment now runs fp32).
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import triton

from zoo.model.resplat.unimatch.cost_volume_triton import (
    cost_volume_kernel,
    cost_volume_triton,
)

_REPO = Path(__file__).resolve().parents[4]
_PIPE = _REPO / "tests/artifacts/resplat/pipeline"

# fixed engine geometry (resplat-base-dl3dv-512x960-view8)
BV, K, C, D, H, W = 8, 2, 128, 128, 64, 120

_SRC_ONNX = _PIPE / "stage0.onnx.orig"
_CV_OUT = "/encoder/depth_predictor/ReduceMean_output_0"
_GRID = "/encoder/depth_predictor/Reshape_37_output_0"
_TGT = "/encoder/depth_predictor/Reshape_30_output_0"
_REF_UNSQUEEZED = "/encoder/depth_predictor/Unsqueeze_166_output_0"


def _misaligned(shape) -> torch.Tensor:
    """Fresh tensor view at a 4-byte offset: warming up the kernel on these
    keeps 16-byte-divisibility OUT of the cubin's baked assumptions, so TRT
    tensors at arbitrary pool offsets index correctly."""
    n = 1
    for d in shape:
        n *= d
    big = torch.empty(n + 4, device="cuda")
    return big[1 : 1 + n].view(shape)


def gen_triton_bundle(out_dir: Path, validate: bool = True) -> Path:
    """Compile the kernel and hand-author the kernel bundle (v2 manifest).

    The manifest declares only the 3 inputs + 1 output — zero intermediate
    buffers — so the TritonReplayPlugin's no-reuse workspace sizing stays
    ~0. ``num_scratch_args: 2`` matches the plugin's launcher contract
    (Triton kernels take trailing hidden scratch pointers).
    """
    import onnx  # noqa: F401  (availability check for validate path)

    ref = torch.randn(BV, C, H, W, device="cuda")
    tgt = torch.randn(BV * K, C, H, W, device="cuda")
    grid = torch.rand(BV * K, D * H, W, 2, device="cuda") * 2 - 1

    compiled = cost_volume_kernel.warmup(
        _misaligned(ref.shape),
        _misaligned(tgt.shape),
        _misaligned(grid.shape),
        _misaligned([BV, D, H, W]),
        K,
        C,
        D,
        H,
        W,
        triton.next_power_of_2(W),
        grid=(1, 1, 1),
    )
    meta = compiled.metadata
    major, minor = torch.cuda.get_device_capability()
    triton.next_power_of_2(W)

    bundle = out_dir / "CostVolume_0.kernels"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "kernel_0000.cubin").write_bytes(compiled.asm["cubin"])

    def entry(name, shape, stride, bid, kind):
        return {
            "name": name,
            "dtype": "float32",
            "shape": shape,
            "stride": stride,
            "buffer_id": bid,
            "kind": kind,
            "source": name,
        }

    shapes = {
        "ref": ([BV, C, H, W], [C * H * W, H * W, W, 1], 0),
        "tgt": ([BV * K, C, H, W], [C * H * W, H * W, W, 1], 1),
        "grid": ([BV * K, D * H, W, 2], [D * H * W * 2, W * 2, 2, 1], 2),
        "out": ([BV, D, H, W], [D * H * W, H * W, W, 1], 3),
    }

    def tensor_arg(name, direction=None):
        shape, stride, bid = shapes[name]
        arg = {
            "kind": "tensor",
            "buffer_id": bid,
            "shape": shape,
            "stride": stride,
            "name": f"buf{bid}",
        }
        if direction:
            arg["direction"] = direction
        return arg

    manifest = {
        "schema_version": 2,
        "module": {
            "type_name": "CostVolume:0",
            "python_class": "cost_volume_triton",
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
        },
        "io": {
            "inputs": [
                {
                    "name": n,
                    "dtype": "float32",
                    "shape": shapes[n][0],
                    "buffer_id": shapes[n][2],
                }
                for n in ("ref", "tgt", "grid")
            ],
            "outputs": [
                {
                    "name": "out",
                    "dtype": "float32",
                    "shape": shapes["out"][0],
                    "buffer_id": 3,
                }
            ],
        },
        "pipeline": [
            {
                "graph": "cv",
                "buffers": {
                    n: entry(n, s[0], s[1], s[2], "allocate" if n == "out" else "input")
                    for n, s in shapes.items()
                },
                "steps": [
                    {
                        "type": "triton_kernel",
                        "kernel": meta.name,
                        "cubin": "kernel_0000.cubin",
                        "device_target": {
                            "backend": "cuda",
                            "arch": major * 10 + minor,
                            "warp_size": 32,
                        },
                        "launch": {
                            "num_warps": meta.num_warps,
                            "num_ctas": 1,
                            "shared_mem_bytes": meta.shared,
                            "num_regs": 0,
                            "grid_expr": None,
                            "captured_grid": [BV, D, H],
                            "num_scratch_args": 2,
                        },
                        "args": [
                            tensor_arg("ref"),
                            tensor_arg("tgt"),
                            tensor_arg("grid"),
                            tensor_arg("out", "out"),
                        ],
                        "output": {"name": "out", "buffer_id": 3},
                    }
                ],
            }
        ],
        "buffers": [
            {
                "id": bid,
                "kind": "input" if n != "out" else "output",
                "dtype": "float32",
                "shape": s[0],
                "stride": s[1],
                "name": f"input_{bid}" if n != "out" else "output_0",
            }
            for n, s, bid in (
                (n, shapes[n], shapes[n][2]) for n in ("ref", "tgt", "grid", "out")
            )
        ],
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if validate:
        from hyperonnx.compile.testing import replay

        with torch.no_grad():
            want = cost_volume_triton(ref, tgt, grid)
            got = replay(bundle, [ref, tgt, grid], cutlass=False)
        diff = (got - want).abs().max().item()
        assert diff < 1e-5, f"bundle replay mismatch: {diff}"
    return bundle


def patch_stage0_triton(src: Path, dst: Path, bundle: Path) -> None:
    """Replace the cost-volume segment with the ``trt::triton`` plugin node.

    The ref input is traced back through the broadcast-unsqueeze chain to
    the 4-D feature tensor; the node produces the original ReduceMean
    output name so downstream wiring is untouched; producers made dead by
    the rewiring are removed (DCE).
    """
    import onnx
    from onnx import helper

    m = onnx.load(str(src), load_external_data=False)
    g = m.graph
    producer = {o: n for n in g.node for o in n.output}

    ref_node = producer[_REF_UNSQUEEZED]
    ref_name = _REF_UNSQUEEZED
    while ref_node.op_type == "Unsqueeze":
        ref_name = ref_node.input[0]
        ref_node = producer[ref_name]

    old = producer[_CV_OUT]
    idx = list(g.node).index(old)
    g.node.remove(old)
    plugin = helper.make_node(
        "triton",
        [ref_name, _TGT, _GRID],
        [_CV_OUT],
        name="cost_volume_triton",
        domain="trt",
        kernels_dir=str(bundle.resolve()),
    )
    g.node.insert(idx, plugin)

    changed = True
    while changed:
        changed = False
        consumed = {i for n in g.node for i in n.input}
        out_names = {o.name for o in g.output}
        for n in list(g.node):
            if not (set(n.output) & (consumed | out_names)):
                g.node.remove(n)
                changed = True

    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


def build_triton_engine(engines_dir: Path, src: Path | None = None) -> None:
    """Bundle + patch + build; stage the engine as stage0.trtbin."""
    engines_dir.mkdir(parents=True, exist_ok=True)
    bundle = gen_triton_bundle(engines_dir / "_build")
    patched = _PIPE / "stage0_triton.onnx"
    patch_stage0_triton(src or _SRC_ONNX, patched, bundle)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "xpu.command.xpu_cli",
            "nv",
            "dump",
            str(patched),
            "--fp16",
            "--dump",
            str(engines_dir),
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"engine build failed:\n{proc.stdout[-2000:]}")
    built = engines_dir / "stage0_triton.trtbin"
    if not built.exists():
        raise FileNotFoundError(f"engine not produced: {built}")
    built.replace(engines_dir / "stage0.trtbin")
    cache = engines_dir / "stage0_triton.timing_cache"
    if cache.exists():
        cache.replace(engines_dir / "stage0.timing_cache")
    step = engines_dir.parent / "stage1_step.trtbin"
    for fallback in sorted(engines_dir.parent.glob("trt-*-fp16")):
        candidate = fallback / "stage1_step.trtbin"
        if candidate.exists():
            step = candidate
            break
    if step.exists() and not (engines_dir / "stage1_step.trtbin").exists():
        shutil.copy(step, engines_dir / "stage1_step.trtbin")
    print(f"stage0 (triton) + step staged in {engines_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "engines_dir",
        type=Path,
        help="target engine directory, e.g. "
        "results/resplat/engines/win32/trt-10.16.0-fp16-triton",
    )
    args = parser.parse_args()
    build_triton_engine(args.engines_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
