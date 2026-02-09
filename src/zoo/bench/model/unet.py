from zoo.model.unet import UNetSeeInDark

from ..registry import BENCH

BENCH.register("unet")(UNetSeeInDark)
