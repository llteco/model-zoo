from zoo.model.aitm import AITM3

from ..registry import BENCH

BENCH.register("aitm")(AITM3)
