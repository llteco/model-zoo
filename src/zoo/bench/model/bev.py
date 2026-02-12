import torch

from zoo.model.bev.projection import LiftSplatShoot

from ..registry import BENCH

grid_config = {
    "x": (-3.0, 3.0, 0.06),
    "y": (0.0, 6.0, 0.06),
    "z": (-1, 3.0, 4.0),
    "depth": (0.4, 6.0, 0.07),
}


@BENCH.register("bev_proj")
class ForwardProjection(LiftSplatShoot):
    """Benchmarking class for LiftSplatShoot module."""

    def __init__(
        self,
        x: tuple[float, float, float] = grid_config["x"],
        y: tuple[float, float, float] = grid_config["y"],
        z: tuple[float, float, float] = grid_config["z"],
        depth: tuple[float, float, float] = grid_config["depth"],
        use_gemm: bool = False,
    ):
        super(ForwardProjection, self).__init__(x, y, z, depth)
        self.cum_gemm = use_gemm

    @property
    def default_inputs(self):
        """Generate default inputs for benchmarking."""
        image_feat = torch.empty(1, 80, 24, 40)
        depth_feat = torch.empty(1, 80, 24, 40)
        return {
            "feat": image_feat,
            "depth": depth_feat,
        }
