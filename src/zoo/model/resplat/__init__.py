"""Vendored ReSplat encoder (simplified for inference/export): the hydra
config machinery is replaced by a dataclass preset, the pointops CUDA kNN by
the exportable ``zoo.model.resplat.knn.KnnQuery`` module, and training-only
code paths are removed.

Stages exposed via ``zoo.export``:
    resplat.stage0           initial Gaussian prediction
    resplat.stage1_step      ONE refinement step (rasterizer external)
    resplat.stage1_state_init, resplat.stage1_gtfeat   helpers
"""

from zoo.model.resplat.encoder_resplat import (
    EncoderReSplat,
    EncoderReSplatCfg,
    GaussianCfg,
)
from zoo.model.resplat.knn import KnnQuery
from zoo.model.resplat.wrappers import (
    Stage0Wrapper,
    Stage1GtFeatures,
    Stage1StateInit,
    Stage1StepWrapper,
)

__all__ = [
    "EncoderReSplat",
    "EncoderReSplatCfg",
    "GaussianCfg",
    "KnnQuery",
    "Stage0Wrapper",
    "Stage1GtFeatures",
    "Stage1StateInit",
    "Stage1StepWrapper",
    "dl3dv_cfg",
    "build_encoder",
]
