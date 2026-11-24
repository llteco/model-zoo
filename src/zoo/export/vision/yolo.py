import os
import pickle
from pathlib import Path

import pooch
import torch
import torch.nn.functional as F
import torchvision as tv

from zoo.model.yolo.yolo26 import YOLO26

from .. import EXPORT

_MODELS = Path(__file__).parents[4] / "models"
# Ultralytics pads with 114/255 to a square of the network stride multiple.
_PAD = 114 / 255

# Official weights: yolo26n.pt from the Ultralytics v8.4.0 asset release (the
# checkpoint the port in ``zoo.model.yolo`` was written against).
_YOLO26N_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
)
_YOLO26N_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
_YOLO26N_FNAME = "yolo26n.pt"
# bare ``{"model": state_dict}`` extracted from the official checkpoint
_YOLO26N_STATE = "yolo26n.state.pt"

# The official checkpoint pickles a full ultralytics model object. Only the
# standard library and torch globals are unpickled for real; every other class
# becomes an inert stub, so ultralytics is never imported and none of its code
# runs. The tensors live in the modules' ``_parameters``/``_buffers`` dicts.
_UNPICKLE_MODULES = {"builtins", "collections", "copyreg", "numpy", "torch"}


class _Stub:
    """Inert stand-in for any non-allowlisted class in the checkpoint."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in _UNPICKLE_MODULES:
            try:
                return super().find_class(module, name)
            except (AttributeError, ImportError):
                return _Stub
        return _Stub


class _PickleModule:
    """``pickle_module`` shim for ``torch.load``."""

    Unpickler = _Unpickler
    load = staticmethod(pickle.load)


def _module_state(obj, prefix: str = "") -> dict[str, torch.Tensor]:
    """Collect ``_parameters``/``_buffers`` recursively from a rebuilt module tree."""
    state = {}
    for name, tensor in (getattr(obj, "_parameters", None) or {}).items():
        state[prefix + name] = tensor
    for name, tensor in (getattr(obj, "_buffers", None) or {}).items():
        state[prefix + name] = tensor
    for name, child in (getattr(obj, "_modules", None) or {}).items():
        state.update(_module_state(child, f"{prefix}{name}."))
    return state


def yolo_weights_dir() -> Path:
    """Cache directory of the YOLO checkpoints (``YOLO_WEIGHTS_DIR``)."""
    cache = os.environ.get("YOLO_WEIGHTS_DIR")
    return Path(cache) if cache else Path(pooch.os_cache("zoo")) / "yolo26n"


def fetch_yolo26n_weights(progressbar: bool = True) -> Path:
    """Download ``yolo26n.pt`` and return the cached bare state dict.

    The official checkpoint is hash-pinned and extracted once into
    ``yolo26n.state.pt`` next to it, so later loads use plain ``torch.load``.
    """
    raw = Path(
        pooch.retrieve(
            url=_YOLO26N_URL,
            known_hash=_YOLO26N_SHA256,
            fname=_YOLO26N_FNAME,
            path=yolo_weights_dir(),
            progressbar=progressbar,
        )
    )
    state_file = raw.parent / _YOLO26N_STATE
    if not state_file.exists():
        checkpoint = torch.load(
            raw, map_location="cpu", pickle_module=_PickleModule, weights_only=False
        )
        torch.save({"model": _module_state(checkpoint["model"])}, state_file)
    return state_file


def letterbox(img: torch.Tensor, size: int) -> torch.Tensor:
    """Resize (C, H, W) float RGB to `size` x `size`, keeping aspect, centred."""
    h, w = img.shape[-2:]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    img = F.interpolate(
        img[None], size=(nh, nw), mode="bilinear", align_corners=False, antialias=True
    )[0]
    out = torch.full((img.shape[0], size, size), _PAD, dtype=img.dtype)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[:, top : top + nh, left : left + nw] = img
    return out


@EXPORT.register("yolo26n")
class ExportYOLO26n(YOLO26):
    """YOLO26n detection: images in, (1, 300, 6) [x1, y1, x2, y2, score, class] out.

    Loads the official ``yolo26n.pt`` by default (downloaded once via pooch);
    ``--checkpoint`` selects a local checkpoint and ``--random_init`` skips the
    weights entirely.
    """

    def __init__(
        self,
        imgsz: int = 640,
        checkpoint: str = "",
        random_init: bool = False,
    ):
        super().__init__(checkpoint=None)
        self.imgsz = imgsz
        if not random_init:
            self.load(Path(checkpoint) if checkpoint else fetch_yolo26n_weights())

    @property
    def default_inputs(self):
        img = tv.io.decode_image(
            str(_MODELS / "vision/siamese.jpg"),
            mode=tv.io.ImageReadMode.RGB,
        )
        img = img.to(torch.float32).div(255.0)
        return {"images": letterbox(img, self.imgsz)[None]}

    @property
    def input_names(self):
        return ["images"]

    @property
    def output_names(self):
        return ["detections"]
