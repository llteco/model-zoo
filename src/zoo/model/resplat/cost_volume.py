"""Weight-free cost-volume computation shared by the model and the server.

The warp + correlation is TensorRT's peak-memory segment in stage0
(~2.4 GiB of tactic scratch at D=128): the fused kernel stages the
[BV*K, C, D, H, W] warped tensor. Running it in torch with depth-candidate
chunking keeps the same math while bounding the peak to a few hundred MiB.
"""

import torch

from zoo.model.resplat.unimatch.matching import warp_with_pose_depth_candidates


def cost_volume(
    ref_features: torch.Tensor,  # [BV, C, H, W]
    tgt_features: torch.Tensor,  # [BV, K, C, H, W]
    intrinsics_input: torch.Tensor,  # [BV*K, 3, 3]
    pose_curr: torch.Tensor,  # [BV, K, 4, 4]
    depth_candidates_curr: torch.Tensor,  # [BV*K, D, H, W]
    *,
    sample_log_depth: bool = True,
    chunk: int | None = None,
) -> torch.Tensor:
    """Warp + correlate + average over source views -> [BV, D, H, W].

    ``chunk`` slices the depth candidates to bound peak memory; every
    candidate's warp and dot are independent so the result is identical.
    """
    b_new, _, c, h, w = tgt_features.shape
    num_depth_candidates = depth_candidates_curr.shape[1]
    chunk = chunk or num_depth_candidates

    ref_features = ref_features.float()
    tgt_features = tgt_features.float()
    depth_candidates_curr = depth_candidates_curr.float()

    chunks = []
    for d0 in range(0, num_depth_candidates, chunk):
        d1 = min(d0 + chunk, num_depth_candidates)
        warped_chunk = warp_with_pose_depth_candidates(
            tgt_features.reshape(-1, c, h, w),
            intrinsics_input.reshape(-1, 3, 3),
            pose_curr.reshape(-1, 4, 4),
            torch.exp(depth_candidates_curr[:, d0:d1])
            if sample_log_depth
            else 1.0 / depth_candidates_curr[:, d0:d1],
        )  # [BV*K, C, d, H, W]
        warped_chunk = warped_chunk.reshape(
            b_new, tgt_features.size(1), *warped_chunk.shape[1:]
        )
        chunks.append(
            (ref_features.unsqueeze(-3).unsqueeze(1) * warped_chunk).sum(2) / (c**0.5)
        )
    # [BV, K, D, H, W] -> [BV, D, H, W], average across the other views
    return torch.cat(chunks, dim=2).mean(1)
