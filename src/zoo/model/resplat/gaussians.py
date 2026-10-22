import torch
from torch import Tensor


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    # rearrange(o, "... (i j) -> ... i j", i=3, j=3)
    return o.reshape(*o.shape[:-1], 3, 3)


def build_covariance(
    scale: Tensor,
    rotation_xyzw: Tensor,
) -> Tensor:
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return rotation @ scale @ scale.transpose(-1, -2) @ rotation.transpose(-1, -2)
