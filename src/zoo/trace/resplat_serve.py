"""ReSplat serving backends for the xpu vision HTTP server.

Self-contained orchestration of the two exported ReSplat TensorRT graphs
(zoo.export resplat.stage0 / resplat.stage1_step; see
zoo.model.resplat.wrappers for the graph contracts)::

    stage0          images + poses -> initial Gaussians + recurrent state
                    + cached GT render-error features (the former
                    state_init / gtfeat graphs are folded in)
    step (x num_refine, rasterizer external)
                    rendered context views -> refined Gaussians + state

Three models are served from these two graphs:

    resplat             composite one-shot: 8 frames + poses -> 3DGS PLY
                        (the rasterizer feedback loop runs server-side)
    resplat-stage0      stage0 alone -> Gaussians + refinement init as
                        tensors, for clients that render/refine themselves
    resplat-step        ONE refinement step; the client supplies its own
                        rendered feedback and carries state between calls

Load with the xpu vision server's registry::

    from xpu.vision.registry import ModelRegistry
    registry = ModelRegistry.from_sources(configs=["zoo.trace.resplat_serve"])

Split-stage protocol (downstream runs its own rasterizer):

    POST /v1/infer resplat-stage0   8 frames + poses
        -> means/scales/rotations/rotations_unnorm/opacities/opacities_raw/
           shs/state/input_view_features   (JSON tensors, big ones base64)
    render the Gaussians client-side (gsplat or any 3DGS rasterizer)
    POST /v1/infer resplat-step     rendered views + carried tensors
        -> refined Gaussians + state_out   (repeat per refinement step)

Carried tensors: every ``resplat-step`` input port accepts either an
inline JSON tensor (``{"shape": [...], "dtype": "float32", "base64": ...}``)
or one shared ``.npz`` multipart part — np.savez of the previous
response's outputs, unchanged: array names are matched against the step
port names AND their aliases from the producing stage (``means`` feeds
``prev_means``, ``state_out`` feeds ``state``). ``rendered_color`` /
``context_image`` additionally accept 8 PNG parts like the ``image`` port
of stage0. ``opacities_raw`` (pre-sigmoid logits) is served so responses
echo back verbatim; ``opacities`` and unit ``rotations`` ride along for
client-side rendering.

The rendered feedback of the composite model is produced by the
xpu-low-level splat renderer (zoo.trace.resplat_render.render_views_gs,
~0.8ms/view; the pure-torch tile EWA rasterizer stays available as the
reference and as the automatic fallback when splat_ffi is unavailable).
RESPLAT_RENDERER=auto (default) | gsplat | torch selects the path. The
final Gaussians are serialized as a standard 3DGS binary PLY.

Quaternion conventions: the model emits ``xyzw`` (scipy order, matching
zoo.model.resplat.gaussians); the 3DGS PLY format expects ``wxyz`` and
the conversion happens right before serialization.

Engines are resolved by ``default_engines_dir()`` ($RESPLAT_ENGINES_DIR,
else the TRT-version-keyed cache under results/resplat/engines/) and
shared: one resident TRT session per graph, refcounted across the three
backends (see ``_acquire_engine``), so the GPU footprint is that of a
single model. stage0 must be the process's first TRT session — the
composite model is registered first and registry setup() acquires in
registration order.
"""
# pylint: disable=too-many-lines,import-outside-toplevel,import-error
# pylint: disable=broad-exception-caught,unused-argument

import io
import os
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from zoo.trace.resplat_render import (
    build_covariance,
    render_view,
    render_views_gs,
    write_gaussian_ply,
)

# renderer selection for the refinement feedback: auto (fast CUDA renderer,
# torch fallback when the extension cannot be built), gsplat, torch
_RENDER_MODE = os.environ.get("RESPLAT_RENDERER", "auto").lower()

# fixed engine geometry (resplat-base-dl3dv-512x960-view8)
V, H, W = 8, 512, 960
LH, LW = H // 4, W // 4
N = V * LH * LW

ENGINES = ("stage0", "stage1_step")

_INPUT_PORTS = ("image", "extrinsics", "intrinsics", "near", "far")
_OUTPUT_PORTS = ("gaussians",)

_REPO = Path(__file__).resolve().parents[4]

_BUILD_HINT = (
    "build them with: uv run pytest tests/resplat/test_resplat_engines.py "
    "-m post (or point RESPLAT_ENGINES_DIR at a directory holding "
    "<name>.trtbin for: " + ", ".join(ENGINES) + ")"
)


def default_engines_dir() -> Path:
    """Resolve the engine directory: $RESPLAT_ENGINES_DIR, else the
    TRT-version-keyed cache under results/resplat/engines/. The fp16
    variant is preferred when present (four fp32 engines exhaust a 16 GB
    card); the fp32 plans stay as the parity reference."""

    env = os.environ.get("RESPLAT_ENGINES_DIR")
    if env:
        return Path(env)
    plat = _REPO / "results" / "resplat" / "engines" / sys.platform
    try:
        from xpu.compiler import runtime_version

        for suffix in ("-fp16", ""):
            current = plat / f"trt-{runtime_version()}{suffix}"
            if current.is_dir():
                return current
    except Exception:  # noqa: BLE001 - fall through to the cache scan
        pass
    if plat.is_dir():
        cached = sorted(plat.glob("trt-*"))
        fp16 = [path for path in cached if path.name.endswith("-fp16")]
        if fp16:
            return fp16[-1]
        if cached:
            return cached[-1]
    return plat / "trt-none"


# ---- shared engine pool ----------------------------------------------------
#
# One resident TRT session per serialized plan, refcounted across every
# backend serving it: the composite model and the split stage0/step models
# share GPU memory instead of loading duplicate engines. A module-wide
# lock serializes GPU work — the HTTP layer may run requests concurrently
# (bounded by its queue depth), and shared sessions must not interleave
# set_inputs/run/get_outputs.

_ENGINES: dict[tuple[str, str], "_EngineRunner"] = {}
_ENGINE_LOCK = threading.Lock()


def _resolve_engine_path(engines_dir: Path, name: str) -> Path:
    trtbin = engines_dir / f"{name}.trtbin"
    onnx = engines_dir / f"{name}.onnx"
    if not onnx.exists():
        onnx = _REPO / "tests" / "artifacts" / "resplat" / "pipeline" / f"{name}.onnx"
    if trtbin.exists():
        return trtbin
    if onnx.exists():
        return onnx  # builds on load; slow but works
    raise FileNotFoundError(
        f"engine {name!r} not found under {engines_dir} (trtbin or onnx). {_BUILD_HINT}"
    )


class _EngineRunner:
    """One resident TRT session for one exported graph (see ENGINES)."""

    def __init__(self, name: str, path: Path, builder_config: str = "") -> None:
        self.name = name
        self.path = path
        self.builder_config = builder_config
        self.refs = 0
        self.session: Any = None
        self.inputs: tuple[str, ...] = ()
        self.dtypes: dict[str, np.dtype] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}

    def load(self) -> None:
        """Create the TRT session and snapshot its input metadata."""
        from xpu.compiler import TrtSession

        # keep the host responsive under the engine loop: cap torch CPU
        # threads at half the cores (mirrors the tests/resplat conventions)
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        self.session = TrtSession(str(self.path), self.builder_config)
        self.inputs = tuple(self.session.get_input_names())
        self.dtypes = {
            port: np.dtype(self.session.get_tensor_dtype(port)) for port in self.inputs
        }
        self.shapes = {
            port: tuple(self.session.get_tensor_shape(port)) for port in self.inputs
        }

    def unload(self) -> None:
        """Drop the session; freeing the last ref frees the GPU engine."""
        self.session = None

    def io(self) -> dict[str, Any]:
        """Engine I/O metadata (port shapes/dtypes) for /v1/models."""
        return {
            "path": str(self.path),
            "inputs": {
                port: {
                    "shape": list(self.shapes[port]),
                    "dtype": str(self.dtypes[port]),
                }
                for port in self.inputs
            },
            "outputs": {
                port: {
                    "shape": list(self.session.get_tensor_shape(port)),
                    "dtype": self.session.get_tensor_dtype(port),
                }
                for port in self.session.get_output_names()
            },
        }

    def run(self, feeds: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Feed the engine; extra feeds (inputs the builder dropped as
        unused, e.g. ``extrinsics_rot`` on the step graph) are skipped."""

        missing = [port for port in self.inputs if port not in feeds]
        if missing:
            raise ValueError(f"engine {self.name}: unbound inputs {missing}")
        prepared = {}
        for port in self.inputs:
            tensor = feeds[port]
            if isinstance(tensor, np.ndarray):
                tensor = np.ascontiguousarray(
                    tensor.astype(self.dtypes[port], copy=False)
                )
                if not tensor.flags.writeable:
                    # e.g. np.frombuffer on a decoded base64 tensor: DLPack
                    # cannot export readonly arrays
                    tensor = tensor.copy()
            elif isinstance(tensor, torch.Tensor):
                tensor = tensor.contiguous()
            _check_shape(self.name, port, tuple(tensor.shape), self.shapes[port])
            prepared[port] = tensor
        with _ENGINE_LOCK:
            self.session.set_inputs(prepared)
            self.session.run()
            return self.session.get_outputs()


def _check_shape(
    engine: str, name: str, actual: tuple[int, ...], expected: tuple[int, ...]
) -> None:
    """Reject tensors the engine cannot consume without silent truncation
    (dynamic ``-1`` dimensions accept any size). Mirrors the framework's
    TrtFfiBackend shape gate so HTTP-fed tensors fail loudly."""

    if len(actual) != len(expected):
        raise ValueError(
            f"engine {engine}: input {name!r} expected rank {len(expected)} "
            f"with shape {list(expected)}, got shape {list(actual)}"
        )
    for axis, (dim, want) in enumerate(zip(actual, expected)):
        if want >= 0 and dim != want:
            raise ValueError(
                f"engine {engine}: input {name!r} expected shape "
                f"{list(expected)}, got {list(actual)} (axis {axis}: {dim} != {want})"
            )


def _acquire_engine(name: str, engines_dir: Path, builder_config: str = ""):
    """Return the shared runner for one graph, loading it on first use.

    Callers must acquire ``stage0`` before any other engine: TensorRT fixes
    some global execution mode (aux worker streams) at the first session of
    the process, and with a small engine loaded first the resident stage0
    graph degrades ~85x (1.2s -> 106s per run, reproduced locally). The
    model registration order (composite first, then the split stages)
    guarantees this at registry setup()."""

    path = _resolve_engine_path(Path(engines_dir), name)
    key = (str(path.resolve()), builder_config)
    runner = _ENGINES.get(key)
    if runner is None:
        runner = _EngineRunner(name, path, builder_config)
        runner.load()
        _ENGINES[key] = runner
    runner.refs += 1
    return runner


def _release_engine(runner: "_EngineRunner") -> None:
    runner.refs -= 1
    if runner.refs <= 0:
        _ENGINES.pop((str(runner.path.resolve()), runner.builder_config), None)
        runner.unload()


# ---- render feedback + PLY serialization ------------------------------------


@torch.no_grad()
def render_context_views(
    means,  # [N, 3]
    scales,  # [N, 3] activated
    rotations_xyzw,  # [N, 4] unit
    harmonics,  # [N, 3, 16] full SH
    opacities,  # [N]
    extrinsics,  # [V, 4, 4] c2w
    intrinsics,  # [V, 3, 3] normalized
) -> torch.Tensor:
    """Render the current Gaussians into every context view -> [1,V,3,H,W].

    Fast path: the CUDA gsplat renderer (degree-0 SH, matching the feedback
    the pure-torch reference produces). Falls back to the reference when the
    extension is unavailable and the mode is auto.

    Inputs may be host numpy arrays or torch tensors (the engine loop keeps
    everything numpy; torch enters only here, on the render path).
    """

    device = torch.device("cuda")

    def to_device(x):
        if torch.is_tensor(x):
            return x.to(device)
        return torch.as_tensor(np.asarray(x), device=device)

    means = to_device(means)
    scales = to_device(scales)
    rotations_xyzw = to_device(rotations_xyzw)
    harmonics = to_device(harmonics)
    opacities = to_device(opacities)
    extrinsics = to_device(extrinsics)
    intrinsics = to_device(intrinsics)

    if _RENDER_MODE != "torch":
        try:
            return render_views_gs(
                means,
                scales,
                rotations_xyzw,
                harmonics,
                opacities,
                extrinsics,
                intrinsics,
                H,
                W,
                bg_color=0.0,  # ReSplat renders feedback on black (dataset cfg)
                sh_degree=0,
            )
        except Exception as exc:
            if _RENDER_MODE == "gsplat":
                raise
            print(
                f"[resplat] fast renderer unavailable ({exc}); "
                "falling back to the pure-torch rasterizer"
            )

    frames = [
        render_view(
            means,
            build_covariance(scales, rotations_xyzw),
            harmonics[..., 0],
            opacities,
            extrinsics[v],
            intrinsics[v],
            H,
            W,
            bg_color=0.0,  # ReSplat renders feedback on black (dataset cfg)
        )
        for v in range(extrinsics.shape[0])
    ]
    return torch.stack(frames, dim=0).unsqueeze(0)  # [1,V,3,H,W]


def ply_bytes(
    means: np.ndarray,  # [N, 3]
    scales: np.ndarray,  # [N, 3] activated
    rotations_xyzw: np.ndarray,  # [N, 4] unit
    harmonics: np.ndarray,  # [N, 3, d_sh]
    opacities: np.ndarray,  # [N] in (0, 1)
) -> bytes:
    """Serialize final Gaussians as a 3DGS binary PLY (wxyz rotations)."""

    from tempfile import mkstemp

    rotations_wxyz = np.roll(rotations_xyzw, 1, axis=-1)  # xyzw -> wxyz
    fd, path = mkstemp(suffix=".ply")
    os.close(fd)
    try:
        write_gaussian_ply(
            path,
            means,
            scales,
            rotations_wxyz,
            harmonics[..., 0],
            harmonics[..., 1:],
            opacities,
        )
        return Path(path).read_bytes()
    finally:
        os.unlink(path)


# ---- composite backend ------------------------------------------------------


class ResplatBackend:
    """Composite vision backend running the two ReSplat TRT engines."""

    def __init__(
        self, engines_dir: Path | str, builder_config: Path | str = ""
    ) -> None:
        self._dir = Path(engines_dir)
        self._builder_config = str(builder_config or "")
        self._runners: dict[str, _EngineRunner] = {}
        self._inputs: dict[str, tuple[str, ...]] = {}
        self._meta: dict[str, Any] = {"engines": {}}
        self._ready = False

    # ---- Backend protocol ----

    def setup(self) -> None:
        """Acquire both engines (stage0 first; see _acquire_engine)."""
        for name in ENGINES:
            runner = _acquire_engine(name, self._dir, self._builder_config)
            self._runners[name] = runner
            self._inputs[name] = runner.inputs
            self._meta["engines"][name] = runner.io()
        self._ready = True

    def teardown(self) -> None:
        """Release both shared engines."""
        self._ready = False
        for runner in self._runners.values():
            _release_engine(runner)
        self._runners.clear()

    def input_names(self):
        """Engine input ports (image + camera geometry)."""
        return _INPUT_PORTS

    def output_names(self):
        """Served output ports (the PLY bytes)."""
        return _OUTPUT_PORTS

    def io_metadata(self) -> dict[str, Any]:
        """Port documentation for /v1/models."""
        meta = dict(self._meta)
        meta["ports"] = {
            "image": f"[1,{V},3,{H},{W}] float32 context frames in [0,1]",
            "extrinsics": f"[1,{V},4,4] float32 c2w (OpenCV convention)",
            "intrinsics": f"[1,{V},3,3] float32 normalized K",
            "near": f"[1,{V}] float32",
            "far": f"[1,{V}] float32",
            "gaussians": f"{N} Gaussians as 3DGS binary PLY",
        }
        return meta

    def health(self) -> dict[str, Any]:
        """Backend liveness + resident engines."""
        return {
            "status": "ok" if self._ready else "unavailable",
            "engines_dir": str(self._dir),
            "engines": {name: name in self._runners for name in ENGINES},
        }

    # ---- inference ----

    def _run(self, name: str, feeds: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Feed one engine; extra feeds (inputs the builder dropped as
        unused, e.g. ``extrinsics_rot`` on the step graph) are skipped."""

        return self._runners[name].run(feeds)

    def run_window(
        self,
        tensors: Mapping[str, np.ndarray],
        num_refine: int = 1,
        ctx: Any = None,
    ) -> dict[str, np.ndarray]:
        """Run one window (8 views) -> window-local Gaussians (host numpy).

        ``tensors`` carries the engine geometry: ``image`` [1,V,3,H,W]
        float32 [0,1], ``extrinsics`` [1,V,4,4] c2w in the window's own
        frame (anchor-normalized by the caller, D1), ``intrinsics``
        [1,V,3,3] normalized, ``near``/``far`` [1,V]. Returns means /
        scales / rotations (unit xyzw) / rotations_unnorm /
        opacities_raw / harmonics / opacities / state, each [1,N,*] in the
        window coordinate frame. ``execute()`` serializes the same output
        to PLY; the windowed reconstructor transforms it with ``to_global``.
        """

        if num_refine < 0:
            raise ValueError(f"num_refine must be >= 0, got {num_refine}")

        # torch-free engine path: every feed stays a host numpy array and
        # the FFI uploads it; torch only enters through the render
        # feedback below (resplat_render). This keeps the torch CUDA
        # context tiny and avoids double-staging big tensors.
        offset = np.asarray([N], dtype=np.int64)
        ext_rot = np.ascontiguousarray(tensors["extrinsics"][:, :, :3, :3])

        # loop-carried state (host numpy between engine calls)
        means: np.ndarray | None = None
        scales = rotations = opacities = harmonics = None
        rotations_unnorm = opacities_raw = state = gt_feat = None
        try:
            # ---- stage 0: initial gaussians + refinement init ----
            mark = time.perf_counter()
            s0 = self._run(
                "stage0",
                {
                    "image": tensors["image"],
                    "extrinsics": tensors["extrinsics"],
                    "intrinsics": tensors["intrinsics"],
                    "near": tensors["near"],
                    "far": tensors["far"],
                },
            )
            if ctx is not None:
                ctx.record("stage0", time.perf_counter() - mark)

            means = s0["means"]  # [1,N,3]
            scales = s0["scales"]  # [1,N,3] activated
            rotations = s0["rotations"]  # [1,N,4] unit (xyzw)
            opacities = s0["opacities"]  # activated, in (0, 1)
            harmonics = s0["harmonics"]  # [1,N,3,16]
            rotations_unnorm = s0["rotations_unnorm"]
            # step engines consume pre-sigmoid logits of shape [1,N,1]
            opacities_raw = _logit(np.clip(opacities, 1e-6, 1 - 1e-6)).reshape(1, N, 1)
            opacities = opacities.reshape(1, N)
            # refinement init folded into the stage0 graph
            state = s0["state"]  # [N, 512]
            gt_feat = s0["input_view_features"]  # [V, 256, LH, LW]

            # ---- refinement loop (rasterizer feedback) ----
            for _ in range(num_refine):
                mark = time.perf_counter()
                rendered = render_context_views(
                    means[0],
                    scales[0],
                    rotations[0],
                    harmonics[0],  # [N,3,16]
                    opacities.reshape(-1),
                    tensors["extrinsics"][0],
                    tensors["intrinsics"][0],
                )
                if ctx is not None:
                    ctx.record("render", time.perf_counter() - mark)

                mark = time.perf_counter()
                step = self._run(
                    "stage1_step",
                    {
                        "rendered_color": rendered,
                        "context_image": tensors["image"],
                        "input_view_features": gt_feat,
                        "state": state,
                        "prev_means": means,
                        "prev_scales": scales,
                        "prev_rotations_unnorm": rotations_unnorm,
                        "prev_opacities_raw": opacities_raw,
                        "prev_shs": harmonics.reshape(1, N, -1),
                        "offset": offset,
                        # extrinsics_rot: exported but unused; the builder
                        # drops it, and _run skips extras defensively
                        "extrinsics_rot": ext_rot,
                    },
                )
                if ctx is not None:
                    ctx.record("step", time.perf_counter() - mark)
                del rendered

                state = step["state_out"]
                means = step["means"]
                scales = step["scales"]
                rotations_unnorm = step["rotations_unnorm"]
                rotations = step["rotations"]
                opacities_raw = step["opacities_raw"]
                harmonics = step["shs"].reshape(1, N, 3, -1)
                opacities = _sigmoid(opacities_raw).reshape(1, N)

            return {
                "means": means,
                "scales": scales,
                "rotations": rotations,
                "rotations_unnorm": rotations_unnorm,
                "opacities_raw": opacities_raw,
                "harmonics": harmonics,
                "opacities": opacities,
                "state": state,
            }
        finally:
            del means, scales, rotations, opacities, harmonics, state, gt_feat
            torch.cuda.empty_cache()

    def execute(
        self, tensors: Mapping[str, np.ndarray], ctx: Any = None
    ) -> dict[str, Any]:
        """Run the full window pipeline; returns {"gaussians": ply bytes}."""
        params: dict[str, Any] = dict(ctx.params) if ctx is not None else {}
        out = self.run_window(
            tensors, num_refine=int(params.get("num_refine", 1)), ctx=ctx
        )
        mark = time.perf_counter()
        ply = ply_bytes(
            out["means"].reshape(N, 3).astype(np.float32),
            out["scales"].reshape(N, 3).astype(np.float32),
            out["rotations"].reshape(N, 4).astype(np.float32),
            out["harmonics"].reshape(N, 3, -1).astype(np.float32),
            out["opacities"].reshape(N).astype(np.float32),
        )
        if ctx is not None:
            ctx.record("ply", time.perf_counter() - mark)
        return {"gaussians": ply}


# ---- split-stage backends ---------------------------------------------------


class ResplatStage0Backend:
    """stage0 alone: images+poses -> initial Gaussians + refinement init.

    Serves the split pipeline (downstream renders and refines itself).
    Outputs are chosen so the client can echo them straight into
    ``resplat-step``: ``opacities_raw`` is the pre-sigmoid logit (computed
    here from the engine's activated opacities) and ``shs`` is the flat
    [1, N, 48] view of the harmonics. ``opacities`` (activated) and the
    unit quaternions ``rotations`` ride along for client-side rendering.
    """

    _OUT = (
        "means",
        "scales",
        "rotations",
        "rotations_unnorm",
        "opacities",
        "opacities_raw",
        "shs",
        "state",
        "input_view_features",
    )

    def __init__(
        self, engines_dir: Path | str, builder_config: Path | str = ""
    ) -> None:
        self._dir = Path(engines_dir)
        self._builder_config = str(builder_config or "")
        self._runner: _EngineRunner | None = None
        self._ready = False

    # ---- Backend protocol ----

    def setup(self) -> None:
        """Acquire the shared stage0 engine."""
        self._runner = _acquire_engine("stage0", self._dir, self._builder_config)
        self._ready = True

    def teardown(self) -> None:
        """Release the shared stage0 engine."""
        self._ready = False
        if self._runner is not None:
            _release_engine(self._runner)
            self._runner = None

    def input_names(self):
        """Engine input ports (image + camera geometry)."""
        return _INPUT_PORTS

    def output_names(self):
        """Served output tensor ports."""
        return self._OUT

    def io_metadata(self) -> dict[str, Any]:
        """Port documentation for /v1/models."""
        return {
            "engines": {"stage0": self._runner.io() if self._runner else {}},
            "ports": {
                "image": f"[1,{V},3,{H},{W}] float32 context frames in [0,1]",
                "extrinsics": f"[1,{V},4,4] float32 c2w (OpenCV convention)",
                "intrinsics": f"[1,{V},3,3] float32 normalized K",
                "near": f"[1,{V}] float32",
                "far": f"[1,{V}] float32",
                "means": f"[1,{N},3] float32 (window frame)",
                "scales": f"[1,{N},3] float32 activated",
                "rotations": f"[1,{N},4] float32 unit quaternions (xyzw)",
                "rotations_unnorm": f"[1,{N},4] float32 raw",
                "opacities": f"[1,{N}] float32 activated",
                "opacities_raw": f"[1,{N},1] float32 pre-sigmoid logits",
                "shs": f"[1,{N},48] float32 (harmonics [N,3,16] flattened)",
                "state": f"[{N},512] float32 recurrent state",
                "input_view_features": f"[{V},256,{LH},{LW}] float32 GT feats",
            },
        }

    def health(self) -> dict[str, Any]:
        """Backend liveness + shared-session refcount."""
        return {
            "status": "ok" if self._ready else "unavailable",
            "engine": "stage0",
            "shared": self._runner.refs if self._runner else 0,
        }

    def execute(
        self, tensors: Mapping[str, np.ndarray], ctx: Any = None
    ) -> dict[str, np.ndarray]:
        """Run stage0 and decode the client-facing output ports."""
        assert self._runner is not None, "setup() was not called"
        mark = time.perf_counter()
        s0 = self._runner.run(
            {
                "image": tensors["image"],
                "extrinsics": tensors["extrinsics"],
                "intrinsics": tensors["intrinsics"],
                "near": tensors["near"],
                "far": tensors["far"],
            }
        )
        if ctx is not None:
            ctx.record("stage0", time.perf_counter() - mark)

        opacities = s0["opacities"].reshape(1, N)  # activated, in (0, 1)
        return {
            "means": s0["means"],
            "scales": s0["scales"],
            "rotations": s0["rotations"],
            "rotations_unnorm": s0["rotations_unnorm"],
            "opacities": opacities,
            # step engines consume pre-sigmoid logits of shape [1,N,1]
            "opacities_raw": _logit(np.clip(opacities, 1e-6, 1 - 1e-6)).reshape(
                1, N, 1
            ),
            "shs": s0["harmonics"].reshape(1, N, -1),
            "state": s0["state"],
            "input_view_features": s0["input_view_features"],
        }


class ResplatStepBackend:
    """stage1_step alone: ONE refinement step, rasterizer external.

    The client supplies its own rendered feedback (``rendered_color``) and
    carries the recurrent state / previous Gaussians from the previous
    call; ``offset`` (the pointops segment end) is supplied internally.
    ``opacities`` (activated) is appended for client-side rendering."""

    _IN = (
        "rendered_color",
        "context_image",
        "input_view_features",
        "state",
        "prev_means",
        "prev_scales",
        "prev_rotations_unnorm",
        "prev_opacities_raw",
        "prev_shs",
    )
    _OUT = (
        "state_out",
        "means",
        "scales",
        "rotations_unnorm",
        "rotations",
        "opacities_raw",
        "shs",
        "opacities",
    )

    def __init__(
        self, engines_dir: Path | str, builder_config: Path | str = ""
    ) -> None:
        self._dir = Path(engines_dir)
        self._builder_config = str(builder_config or "")
        self._runner: _EngineRunner | None = None
        self._ready = False

    # ---- Backend protocol ----

    def setup(self) -> None:
        """Acquire the shared step engine."""
        self._runner = _acquire_engine("stage1_step", self._dir, self._builder_config)
        self._ready = True

    def teardown(self) -> None:
        """Release the shared step engine."""
        self._ready = False
        if self._runner is not None:
            _release_engine(self._runner)
            self._runner = None

    def input_names(self):
        """Client-facing input ports (offset is supplied internally)."""
        return self._IN

    def output_names(self):
        """Served output tensor ports."""
        return self._OUT

    def io_metadata(self) -> dict[str, Any]:
        """Port documentation for /v1/models."""
        return {
            "engines": {"stage1_step": self._runner.io() if self._runner else {}},
            "ports": {
                "rendered_color": f"[1,{V},3,{H},{W}] float32 client render",
                "context_image": f"[1,{V},3,{H},{W}] float32 GT frames",
                "input_view_features": f"[{V},256,{LH},{LW}] (from stage0)",
                "state": f"[{N},512] (stage0 'state' / prior 'state_out')",
                "prev_means": f"[1,{N},3] (stage0/step 'means')",
                "prev_scales": f"[1,{N},3] (stage0/step 'scales')",
                "prev_rotations_unnorm": f"[1,{N},4] ('rotations_unnorm')",
                "prev_opacities_raw": f"[1,{N},1] ('opacities_raw')",
                "prev_shs": f"[1,{N},48] ('shs')",
                "state_out": f"[{N},512] next recurrent state",
                "means": f"[1,{N},3] refined",
                "scales": f"[1,{N},3] refined",
                "rotations_unnorm": f"[1,{N},4] refined raw",
                "rotations": f"[1,{N},4] refined unit (xyzw)",
                "opacities_raw": f"[1,{N},1] refined logits",
                "shs": f"[1,{N},48] refined",
                "opacities": f"[1,{N}] activated",
            },
        }

    def health(self) -> dict[str, Any]:
        """Backend liveness + shared-session refcount."""
        return {
            "status": "ok" if self._ready else "unavailable",
            "engine": "stage1_step",
            "shared": self._runner.refs if self._runner else 0,
        }

    def execute(
        self, tensors: Mapping[str, np.ndarray], ctx: Any = None
    ) -> dict[str, np.ndarray]:
        """Run one refinement step on the client-supplied feedback."""
        assert self._runner is not None, "setup() was not called"
        mark = time.perf_counter()
        out = self._runner.run({**tensors, "offset": np.asarray([N], dtype=np.int64)})
        if ctx is not None:
            ctx.record("step", time.perf_counter() - mark)
        out["opacities"] = _sigmoid(out["opacities_raw"]).reshape(1, N)
        return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p) - np.log1p(-p)


# ---- request preprocessing ---------------------------------------------------


def _prepare_frames(data) -> np.ndarray:
    """Merged image payloads -> [1, V, 3, H, W] float32 in [0, 1]."""

    from PIL import Image

    frames = data if isinstance(data, list) else [data]
    if len(frames) != V:
        raise ValueError(
            f"resplat expects exactly {V} context frames (engines fix V={V}), "
            f"got {len(frames)}"
        )
    prepared = []
    for index, frame in enumerate(frames):
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"frame {index}: expected HWC RGB, got {array.shape}")
        if array.shape[1] != W or array.shape[0] != H:
            array = np.asarray(
                Image.fromarray(array).resize((W, H), Image.Resampling.BILINEAR)
            )
        prepared.append(array.transpose(2, 0, 1).astype(np.float32) / 255.0)
    return np.ascontiguousarray(np.stack(prepared)[np.newaxis])


def _prepare_extrinsics(data) -> np.ndarray:
    from xpu.vision.media import decode_tensor

    ext = decode_tensor(data).astype(np.float32)
    if ext.shape == (4, 4):  # single shared pose -> broadcast
        ext = np.broadcast_to(ext, (V, 4, 4))
    if ext.shape != (V, 4, 4):
        raise ValueError(
            f"extrinsics must be [4,4] or [{V},4,4] c2w, got shape {ext.shape}"
        )
    return ext[np.newaxis].copy()


def _prepare_intrinsics(data) -> np.ndarray:
    from xpu.vision.media import decode_tensor

    k = decode_tensor(data).astype(np.float32)
    if k.shape == (3, 3):
        k = np.broadcast_to(k, (V, 3, 3))
    if k.shape != (V, 3, 3):
        raise ValueError(
            f"intrinsics must be [3,3] or [{V},3,3] (normalized by W,H), "
            f"got shape {k.shape}"
        )
    return k[np.newaxis].copy()


def stage0_preprocess(inputs, ctx):
    """images (8 merged frames) + poses -> stage0 engine feeds."""

    params = ctx.params
    near = float(params.get("near", 0.01))
    far = float(params.get("far", 200.0))
    if not 0 < near < far:
        raise ValueError(f"need 0 < near < far, got near={near} far={far}")
    return {
        "image": _prepare_frames(inputs["image"].data),
        "extrinsics": _prepare_extrinsics(inputs["extrinsics"].data),
        "intrinsics": _prepare_intrinsics(inputs["intrinsics"].data),
        "near": np.full((1, V), near, dtype=np.float32),
        "far": np.full((1, V), far, dtype=np.float32),
    }


#: step input port <- accepted npz array names (step names first, then the
#: producing stage's output names, so responses echo back verbatim)
_ALIASES = {
    "state": ("state", "state_out"),
    "prev_means": ("prev_means", "means"),
    "prev_scales": ("prev_scales", "scales"),
    "prev_rotations_unnorm": ("prev_rotations_unnorm", "rotations_unnorm"),
    "prev_opacities_raw": ("prev_opacities_raw", "opacities_raw"),
    "prev_shs": ("prev_shs", "shs"),
    "input_view_features": ("input_view_features",),
    "rendered_color": ("rendered_color",),
    "context_image": ("context_image",),
}


def _npz_arrays(data: bytes, cache: dict) -> dict[str, np.ndarray]:
    """Load (and memoize, per request) one .npz carry part."""

    if bytes(data[:2]) != b"PK":
        raise ValueError(
            "tensor 'part' payloads must be .npz (np.savez) files; inline "
            "tensors go in the JSON body as {shape, dtype, base64}"
        )
    key = id(data)
    if key not in cache:
        with np.load(io.BytesIO(data)) as archive:  # own data, no pickles
            cache[key] = {name: archive[name] for name in archive.files}
    return cache[key]


def step_preprocess(inputs, ctx):
    """Build the step engine feeds: PNG frames, inline JSON tensors or one
    shared npz carry part (array names = port names or their aliases)."""

    from xpu.vision.media import decode_tensor

    feeds: dict[str, np.ndarray] = {}
    npz_cache: dict = {}
    for port, payload in inputs.items():
        data = payload.data
        if payload.kind == "image":
            feeds[port] = _prepare_frames(data)  # [1,V,3,H,W] float32
        elif isinstance(data, (bytes, bytearray, memoryview)):
            archive = _npz_arrays(bytes(data), npz_cache)
            names = [name for name in _ALIASES.get(port, (port,)) if name in archive]
            if not names:
                raise ValueError(
                    f"npz part for port {port!r} holds none of {sorted(archive)}"
                )
            feeds[port] = np.ascontiguousarray(archive[names[0]])
        else:
            feeds[port] = decode_tensor(data)
    return feeds


# ---- vision server configuration ---------------------------------------------


def build_models(engines_dir: str | os.PathLike | None = None) -> list:
    """The three ModelSpecs for the xpu vision server.

    Serve them with ``ModelRegistry.from_sources(
    configs=["zoo.trace.resplat_serve"])`` (the module's lazy ``MODELS``
    attribute calls this) — xpu.vision is imported here, not at module
    level, so importing this module never requires the serving stack.
    """

    from xpu.vision import (
        ImageSpec,
        InputDecl,
        ModelSpec,
        file_output,
        json_output,
        tensor_input,
    )

    engines = Path(engines_dir) if engines_dir else default_engines_dir()

    stage0_inputs = {
        "image": InputDecl(
            type="image",
            image=ImageSpec(resize=(W, H)),
            description=(
                f"{V} context frames (png/jpg/bmp) in temporal order; "
                "repeated inputs on this port merge into the clip"
            ),
        ),
        "extrinsics": tensor_input(
            description=f"camera-to-world [4,4] or [{V},4,4] (OpenCV/COLMAP)",
        ),
        "intrinsics": tensor_input(
            description="normalized K [3,3] or [V,3,3] (fx,fy,cx,cy in 0..1)",
        ),
    }
    stage0_outputs = {
        "means": "initial Gaussian centers [1,N,3] (window frame)",
        "scales": "activated scales [1,N,3]",
        "rotations": "unit quaternions [1,N,4] (xyzw) for rendering",
        "rotations_unnorm": "raw quaternions [1,N,4] (echo into resplat-step)",
        "opacities": "activated opacities [1,N] for rendering",
        "opacities_raw": "pre-sigmoid logits [1,N,1] (echo into resplat-step)",
        "shs": "spherical harmonics [1,N,48] = [N,3,16] flattened",
        "state": f"recurrent state [{N},512] (echo into resplat-step)",
        "input_view_features": f"cached GT features [{V},256,{LH},{LW}]",
    }
    step_inputs = {
        "rendered_color": (
            f"[1,{V},3,512,960] your rasterizer's render of the current "
            "Gaussians (8 PNG parts or a float32 tensor)"
        ),
        "context_image": (
            f"[1,{V},3,512,960] the original context frames (8 PNG parts or a "
            "float32 tensor)"
        ),
        "input_view_features": "carried from stage0",
        "state": "carried from stage0 'state' or the previous 'state_out'",
        "prev_means": "carried from stage0/step 'means'",
        "prev_scales": "carried from stage0/step 'scales'",
        "prev_rotations_unnorm": "carried from 'rotations_unnorm'",
        "prev_opacities_raw": "carried from 'opacities_raw'",
        "prev_shs": "carried from 'shs'",
    }
    step_outputs = {
        "state_out": "next recurrent state (echo on the next call)",
        "means": "refined centers [1,N,3]",
        "scales": "refined scales [1,N,3]",
        "rotations_unnorm": "refined raw quaternions [1,N,4]",
        "rotations": "refined unit quaternions [1,N,4] (xyzw)",
        "opacities_raw": "refined logits [1,N,1]",
        "shs": "refined harmonics [1,N,48]",
        "opacities": "activated opacities [1,N] for rendering",
    }

    return [
        ModelSpec(
            id="resplat",
            engine=engines,
            backend=lambda spec: ResplatBackend(spec.engine),
            inputs=stage0_inputs,
            outputs={
                "gaussians": file_output(
                    fmt="ply",
                    mime="application/octet-stream",
                    description="3D Gaussians as a standard 3DGS binary PLY",
                ),
            },
            preprocess=stage0_preprocess,
        ),
        ModelSpec(
            id="resplat-stage0",
            engine=engines,
            backend=lambda spec: ResplatStage0Backend(spec.engine),
            inputs=stage0_inputs,
            outputs={port: json_output(desc) for port, desc in stage0_outputs.items()},
            preprocess=stage0_preprocess,
        ),
        ModelSpec(
            id="resplat-step",
            engine=engines,
            backend=lambda spec: ResplatStepBackend(spec.engine),
            inputs={port: tensor_input(desc) for port, desc in step_inputs.items()},
            outputs={port: json_output(desc) for port, desc in step_outputs.items()},
            preprocess=step_preprocess,
        ),
    ]


def __getattr__(name: str):
    """Lazy ``MODELS`` for the vision registry's config loading (xpu.vision
    is only importable where the vision server actually runs)."""

    if name == "MODELS":
        return build_models()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
