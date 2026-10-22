from dataclasses import dataclass

from torch import Tensor


@dataclass
class Gaussians:
    means: Tensor
    covariances: Tensor | None
    harmonics: Tensor
    opacities: Tensor
    scales: Tensor | None = None
    rotations: Tensor | None = None
    probabilities: Tensor | None = None
    mask: Tensor | None = None
    filter_3D: Tensor | None = None
    rotations_unnorm: Tensor | None = None
    scale_factor: Tensor | None = None
    shift: Tensor | None = None
