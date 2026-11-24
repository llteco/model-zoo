import argparse
import os
from pathlib import Path

import pooch
import torch
from torch import nn

from zoo.model.polyroom import (
    DeformableTransformer,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    DeformableTransformerEncoder,
    DeformableTransformerEncoderLayer,
    MSDeformAttn,
    RoomFormer,
    build_polyroom,
)

from .. import EXPORT

# Official checkpoint published with the paper (Google Drive folder linked from
# https://github.com/3dv-casia/PolyRoom#checkpoints). The folder holds
# PolyRoom.pth, Mask2former.pth and mask2former_config.py; only PolyRoom.pth is
# needed here because the instance-segmentation front-end is an input of this
# model (``room_inial``).
_POLYROOM_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1WtFaur3ssGLBd-JABkVnEdGy9qJIFlxI&export=download&confirm=t"
)
_POLYROOM_SHA256 = "aa37ceb8dbfc8e2ea93788180d408c26624fd2959bc88944a40318f643d9aa10"
_POLYROOM_FNAME = "PolyRoom.pth"


def polyroom_weights_dir() -> Path:
    """Cache directory of the official checkpoint (``POLYROOM_WEIGHTS_DIR``)."""
    cache = os.environ.get("POLYROOM_WEIGHTS_DIR")
    return Path(cache) if cache else Path(pooch.os_cache("zoo")) / "polyroom"


def fetch_polyroom_weights(progressbar: bool = True) -> Path:
    """Download the official PolyRoom checkpoint into the pooch cache.

    The download is verified against a pinned SHA-256 and reused on later runs.
    """
    return Path(
        pooch.retrieve(
            url=_POLYROOM_URL,
            known_hash=_POLYROOM_SHA256 or None,
            fname=_POLYROOM_FNAME,
            path=polyroom_weights_dir(),
            progressbar=progressbar,
        )
    )


def read_state_dict(checkpoint: str | os.PathLike) -> dict[str, torch.Tensor]:
    """Load a PolyRoom state dict (accepts the official ``{"model": ...}`` wrapper)."""
    # the official file also pickles the training argparse.Namespace; allowlist
    # just that class so the rest is still loaded with weights_only=True
    with torch.serialization.safe_globals([argparse.Namespace]):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    assert isinstance(state, dict)
    return {key.removeprefix("module."): value for key, value in state.items()}


@EXPORT.register("polyroom")
class ExportPolyRoom(nn.Module):
    """PolyRoom floorplan reconstruction: density map in, polygons out.

    Inputs:
        image: ``(B, 1, H, W)`` top-down point-density map scaled to ``[0, 1]``;
            ``H`` and ``W`` must be multiples of 64 (256 in the paper).
        room_inial: ``(B, num_queries, 2)`` room-query init corners normalised to
            ``[0, 1]``, i.e. the concatenated corners of the instance-segmentation
            polygons padded to ``num_queries``.
        room_inial_mask: ``(B, num_queries)``, 1 for valid ``room_inial`` rows.
            All zeros disables the room-aware initialisation.

    Outputs:
        pred_logits: ``(B, num_polys, num_queries_per_poly)`` corner validity logits.
        pred_coords: ``(B, num_polys, num_queries_per_poly, 2)`` normalised corners.

    .. note::
        Export at ``--opset-version 17`` (the default 23 is rejected below).
        Opset 20 renamed the ONNX ``GridSample`` mode strings (``bilinear`` ->
        ``linear``); TensorRT 10.16 still implements the opset-16 dialect, so a
        opset-20+ graph parses but samples with the wrong mode and silently
        produces different numbers.
    """

    hier = [
        MSDeformAttn,
        DeformableTransformerEncoderLayer,
        DeformableTransformerDecoderLayer,
        DeformableTransformerEncoder,
        DeformableTransformerDecoder,
        DeformableTransformer,
    ]

    def __init__(
        self,
        image_size: int = 256,
        checkpoint: str = "",
        random_init: bool = False,
    ):
        """Loads the official checkpoint by default.

        Args:
            image_size: square input resolution, a multiple of 64.
            checkpoint: explicit checkpoint path; overrides the download.
            random_init: skip the weights entirely (no download, random init);
                used by the export tests, which compare eager and TRT numerics
                on the same fixed-seed weights.
        """
        super().__init__()
        if image_size % 64:
            raise ValueError("image_size must be a multiple of 64")
        self.model: RoomFormer = build_polyroom()
        self.image_size = image_size
        self.eval()
        if not random_init:
            source = Path(checkpoint) if checkpoint else fetch_polyroom_weights()
            self.model.load_state_dict(read_state_dict(source))

    def train(self, mode: bool = True) -> "ExportPolyRoom":
        """Keep dropout disabled; the ONNX tracer switches modules to train mode."""
        return super().train(False)

    def post_process(self, onnx_file, *, opset_version=None, **kwargs):
        """Reject opsets whose ``GridSample`` dialect TensorRT mis-parses."""
        if opset_version is not None and opset_version >= 20:
            Path(onnx_file).unlink(missing_ok=True)
            raise ValueError(
                f"PolyRoom must be exported with --opset-version 17, got "
                f"{opset_version}: opset 20 renamed GridSample's mode strings and "
                f"TensorRT samples with the wrong mode (see class docstring)."
            )

    def forward(
        self,
        image: torch.Tensor,
        room_inial: torch.Tensor,
        room_inial_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return self.model(image, room_inial, room_inial_mask)

    @property
    def default_inputs(self):
        return {
            "image": torch.zeros(1, 1, self.image_size, self.image_size),
            "room_inial": torch.zeros(1, self.model.num_queries, 2),
            "room_inial_mask": torch.zeros(1, self.model.num_queries),
        }

    @property
    def input_names(self):
        return ["image", "room_inial", "room_inial_mask"]

    @property
    def output_names(self):
        return ["pred_logits", "pred_coords"]
