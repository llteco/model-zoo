"""Manual bilinear sampling (grid_sample equivalent) in pure torch ops.

``F.grid_sample`` lowers to a vendor cuDNN kernel, which prevents a pure
Triton kernel bundle. This version uses only gather + arithmetic so inductor
can lower the whole cost volume to Triton. Semantics match
``F.grid_sample(mode="bilinear", padding_mode="zeros", align_corners=True)``.
"""

import torch


def grid_sample_manual(feature: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """feature [B, C, H, W], grid [B, D, H*W, 2] (x,y in [-1,1]) -> [B, C, D, H*W]."""
    b, c, h, w = feature.shape
    x = (grid[..., 0] + 1) * (w - 1) / 2
    y = (grid[..., 1] + 1) * (h - 1) / 2

    x0, y0 = torch.floor(x), torch.floor(y)
    x1, y1 = x0 + 1, y0 + 1

    def clip(v, m):
        return v.clamp(0, m - 1)

    x0c, x1c = clip(x0, w), clip(x1, w)
    y0c, y1c = clip(y0, h), clip(y1, h)
    flat = feature.flatten(2)  # [B, C, H*W]

    def corner(xi, yi):
        idx = (yi * w + xi).long().reshape(b, 1, -1)
        return torch.gather(flat, 2, idx.expand(-1, c, -1))  # [B, C, D*HW]

    v00 = corner(x0c, y0c)
    v01 = corner(x0c, y1c)
    v10 = corner(x1c, y0c)
    v11 = corner(x1c, y1c)

    def valid(xi, yi):
        inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        return inside.to(feature.dtype).reshape(b, 1, -1)

    v00 = v00 * valid(x0, y0)
    v01 = v01 * valid(x0, y1)
    v10 = v10 * valid(x1, y0)
    v11 = v11 * valid(x1, y1)

    wx0, wx1 = (x1 - x).reshape(b, 1, -1), (x - x0).reshape(b, 1, -1)
    wy0, wy1 = (y1 - y).reshape(b, 1, -1), (y - y0).reshape(b, 1, -1)
    return (
        v00 * wx0 * wy0 + v01 * wx0 * wy1 + v10 * wx1 * wy0 + v11 * wx1 * wy1
    ).reshape(b, c, grid.shape[1], grid.shape[2])


if __name__ == "__main__":
    torch.manual_seed(0)
    for b, c, h, w, d in [(2, 4, 8, 10, 6), (16, 128, 64, 120, 16)]:
        feat = torch.randn(b, c, h, w, device="cuda")
        grid = torch.rand(b, d, h * w, 2, device="cuda") * 2.4 - 1.2
        ref = torch.nn.functional.grid_sample(
            feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        got = grid_sample_manual(feat, grid)
        diff = (ref - got).abs().max().item()
        print(f"B={b} C={c}: max abs diff = {diff:.3e}")
