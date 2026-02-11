from zoo.model.raft import RAFT

from ..registry import BENCH

BENCH.register("raft")(RAFT)
