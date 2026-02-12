import torch

from zoo.model.bev.projection import LiftSplatShoot

from .. import EXPORT

grid_config = {
    "x": (-3.0, 3.0, 0.06),
    "y": (0.0, 6.0, 0.06),
    "z": (-1, 3.0, 4.0),
    "depth": (0.4, 6.0, 0.07),
}


@EXPORT.register("bev.projection")
class ExportBEVProjection(LiftSplatShoot):
    """Exportable BEV Projection Module."""

    def __init__(
        self,
        x: tuple[float, float, float] = grid_config["x"],
        y: tuple[float, float, float] = grid_config["y"],
        z: tuple[float, float, float] = grid_config["z"],
        depth: tuple[float, float, float] = grid_config["depth"],
        use_gemm: bool = False,
    ):
        super().__init__(x, y, z, depth)
        self.cum_gemm = use_gemm

    def forward(self, feat: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Forward pass for BEV Projection."""
        feat = feat.view(-1, *feat.shape[2:])
        depth = depth.view(-1, *depth.shape[2:])
        return super().forward(feat, depth)

    @property
    def default_inputs(self):
        """Generate default inputs for benchmarking."""
        image_feat = torch.empty(1, 1, 80, 24, 40)
        depth_feat = torch.empty(1, 1, 80, 24, 40)
        return {
            "feat": image_feat,
            "depth": depth_feat,
        }

    @property
    def input_names(self):
        return ["feat", "depth"]

    @property
    def output_names(self):
        return ["proj"]
