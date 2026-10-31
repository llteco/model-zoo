"""ReSplat exports (stage-0 with refinement init + stage-1 delta-G step).

The rasterizer is external: ``resplat.stage1_step`` takes rendered context
images as an input. The refinement init (recurrent state + cached GT
render-error features) is folded into ``resplat.stage0``. The pointops kNN
is exported directly as ``trt::KnnQuery`` plugin nodes (see
zoo.model.resplat.knn); post-process only declares the ``trt`` opset
import for the custom domain.

Examples:
    uv -m zoo.export resplat.stage0 --cuda -d out/resplat
    uv -m zoo.export resplat.stage1_step --cuda -d out/resplat
"""
# pylint: disable=import-outside-toplevel,import-error,super-init-not-called

import torch

from zoo.export import EXPORT
from zoo.model.resplat.export_compat import install_export_compat
from zoo.model.resplat.presets import build_encoder
from zoo.model.resplat.wrappers import (
    Stage0Wrapper,
    Stage1StepWrapper,
)

V, H, W = 8, 512, 960
LH, LW = H // 4, W // 4
N = V * LH * LW
D_SH = 16


install_export_compat()


def _declare_trt_opset(onnx_path, **_) -> None:
    """Declare the trt opset import for the KnnQuery plugin nodes."""
    import onnx

    model = onnx.load(onnx_path)
    n = sum(1 for node in model.graph.node if node.domain == "trt")
    if n and not any(i.domain == "trt" for i in model.opset_import):
        model.opset_import.append(onnx.helper.make_opsetid("trt", 1))
        onnx.save(model, onnx_path)
    print(f"post-process: {n} trt::KnnQuery node(s)")


class _ResplatExportMixin:
    """Checkpoint resolution shared by the export registrations."""

    @staticmethod
    def _load_encoder(checkpoint: str, num_refine: int):
        """Default to the HF-hosted pretrained checkpoint (verified cache)."""
        from zoo.model.resplat.weights import ensure_checkpoint

        if not checkpoint:
            checkpoint = str(ensure_checkpoint())
        return build_encoder(num_refine=num_refine, checkpoint=checkpoint)

    def __init__(self, checkpoint: str = "", num_refine: int = 1) -> None:
        # cooperative super: the concrete wrapper base takes the encoder
        super().__init__(self._load_encoder(checkpoint, num_refine))  # pyright: ignore[reportCallIssue]

    def post_process(self, f, **_):
        """Declare the trt opset on the exported graph."""
        _declare_trt_opset(f)


def _pinhole(v: int) -> torch.Tensor:
    k = torch.eye(3).repeat(1, v, 1, 1)
    k[..., 0, 0] = k[..., 1, 1] = 0.7
    k[..., 0, 2] = k[..., 1, 2] = 0.5
    return k


def _poses(v: int) -> torch.Tensor:
    ext = torch.eye(4).repeat(1, v, 1, 1)
    ext[..., :3, 3] = torch.randn(1, v, 3) * 0.2
    return ext


@EXPORT.register("resplat.stage0")
class ExportResplatStage0(_ResplatExportMixin, Stage0Wrapper):
    """images+poses -> initial Gaussians + refinement init (state, GT feats)."""

    input_names = ["image", "extrinsics", "intrinsics", "near", "far"]
    output_names = [
        "means",
        "covariances",
        "harmonics",
        "opacities",
        "scales",
        "rotations",
        "rotations_unnorm",
        "state",
        "input_view_features",
    ]

    @property
    def default_inputs(self):
        """Deterministic sample feeds for the export CLI."""
        return {
            "image": torch.rand(1, V, 3, H, W),
            "extrinsics": _poses(V),
            "intrinsics": _pinhole(V),
            "near": torch.full((1, V), 0.01),
            "far": torch.full((1, V), 200.0),
        }


@EXPORT.register("resplat.stage1_step")
class ExportResplatStep(_ResplatExportMixin, Stage1StepWrapper):
    """One refinement step with the rasterizer cut out (rendered views in)."""

    def __init__(self, checkpoint: str = "", num_refine: int = 1) -> None:
        # Stage1StepWrapper also takes (v, image_shape), which the mixin's
        # cooperative super() chain cannot supply
        Stage1StepWrapper.__init__(
            self, self._load_encoder(checkpoint, num_refine), V, (H, W)
        )

    @property
    def default_inputs(self):
        """Deterministic sample feeds for the export CLI."""
        return {
            "rendered_color": torch.rand(1, V, 3, H, W),
            "context_image": torch.rand(1, V, 3, H, W),
            "input_view_features": torch.rand(V, 256, LH, LW),
            "state": torch.rand(N, 512),
            "prev_means": torch.rand(1, N, 3),
            "prev_scales": torch.rand(1, N, 3) * 0.05 + 0.01,
            "prev_rotations_unnorm": torch.rand(1, N, 4),
            "prev_opacities_raw": torch.rand(1, N, 1),
            "prev_shs": torch.rand(1, N, 3 * D_SH) * 0.1,
            "extrinsics_rot": torch.rand(1, V, 3, 3),
            "offset": torch.tensor([N], dtype=torch.int64),
        }

    input_names = [
        "rendered_color",
        "context_image",
        "input_view_features",
        "state",
        "prev_means",
        "prev_scales",
        "prev_rotations_unnorm",
        "prev_opacities_raw",
        "prev_shs",
        "extrinsics_rot",
        "offset",
    ]
    output_names = [
        "state_out",
        "means",
        "scales",
        "rotations_unnorm",
        "rotations",
        "opacities_raw",
        "shs",
    ]
