import torch

from zoo.model.nav_rl import Agent

from ..registry import BENCH


@BENCH.register("navrl")
class BenchNavRL(Agent):
    @property
    def default_inputs(self):
        return {
            "robot_state": torch.zeros([1, 8]),
            "static_obs_input": torch.zeros([1, 1, 36, 4]),
            "dyn_obs_input": torch.zeros([1, 1, 5, 10]),
            "target_dir": torch.zeros([1, 1, 3]),
        }
