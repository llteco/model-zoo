import torch
from torch import Tensor


def homogenize_points(points: Tensor) -> Tensor:
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(
    vectors: Tensor,
) -> Tensor:
    """Convert batched vectors (xyz) to (xyz0)."""
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(
    homogeneous_coordinates: Tensor,
    transformation: Tensor,
) -> Tensor:
    """Apply a rigid-body transformation to points or vectors."""
    # einsum(transformation, homogeneous_coordinates, "... i j, ... j -> ... i");
    # the unsqueeze/squeeze treats the coordinates as a column vector so matmul
    # broadcasts the batch dimensions exactly like the einsum ellipsis did.
    return (transformation @ homogeneous_coordinates.unsqueeze(-1)).squeeze(-1)


def transform_cam2world(
    homogeneous_coordinates: Tensor,
    extrinsics: Tensor,
) -> Tensor:
    """Transform points from 3D camera coordinates to 3D world coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics)


def transform_world2cam(
    homogeneous_coordinates: Tensor,
    extrinsics: Tensor,
) -> Tensor:
    """Transform points from 3D world coordinates to 3D camera coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics.inverse())


def unproject(
    coordinates: Tensor,
    z: Tensor,
    intrinsics: Tensor,
) -> Tensor:
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    coordinates = homogenize_points(coordinates)
    # einsum(intrinsics.inverse(), coordinates, "... i j, ... j -> ... i")
    ray_directions = (intrinsics.inverse() @ coordinates.unsqueeze(-1)).squeeze(-1)

    # Apply the supplied depth values.
    return ray_directions * z[..., None]


def get_world_rays(
    coordinates: Tensor,
    extrinsics: Tensor,
    intrinsics: Tensor,
) -> tuple[
    Tensor,  # origins
    Tensor,  # directions
]:
    # Get camera-space ray directions.
    directions = unproject(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )
    directions = directions / directions[..., -1:]

    # Transform ray directions to world coordinates.
    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, extrinsics)[..., :-1]

    # Tile the ray origins to have the same shape as the ray directions.
    origins = extrinsics[..., :-1, -1].broadcast_to(directions.shape)

    return origins, directions


def sample_image_grid(
    shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> tuple[
    Tensor,  # float coordinates (xy indexing)
    Tensor,  # integer indices (ij indexing)
]:
    """Get normalized (range 0 to 1) coordinates and integer indices for an image."""

    # Each entry is a pixel-wise integer coordinate. In the 2D case, each entry is a
    # (row, col) coordinate.
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)

    # Each entry is a floating-point coordinate in the range (0, 1). In the 2D case,
    # each entry is an (x, y) coordinate.
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)

    return coordinates, stacked_indices
