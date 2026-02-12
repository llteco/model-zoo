from pathlib import Path

import torch
from torch.nn import Module


def _gen_dx_bx(xbound, ybound, zbound):
    dx = [row[2] for row in [xbound, ybound, zbound]]
    bx = [row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]]
    nx = [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    return dx, bx, nx


class LiftSplatShoot(Module):
    def __init__(self, x, y, z, depth):
        super().__init__()
        self.grid_conf = dict(x=x, y=y, z=z, depth=depth)
        dx, bx, nx = _gen_dx_bx(x, y, z)
        self.dx = torch.tensor(dx)
        self.bx = torch.tensor(bx)
        self.nx = nx
        if (Path(__file__).parent / "cam_params.pt").exists():
            rots, trans, intrins, post_rot, post_trans = torch.load(
                Path(__file__).parent / "cam_params.pt", map_location="cpu"
            )
        else:
            rots = torch.eye(3)[None, None]
            trans = torch.zeros([1, 1, 3])
            intrins = torch.tensor([[320, 0, 320], [0, 320, 180], [0, 0, 1]])[
                None, None
            ]
            post_rot = torch.eye(3)[None, None]
            post_trans = torch.zeros([1, 1, 3])
        self.downsample = 16
        self.frustum = self._create_frustum()
        self.geom, self.ranks, self.kept2, self.sorts_t, self.lens = self._get_geometry(
            rots, trans, intrins, post_rot, post_trans
        )
        self.mat = self._gen_mat()
        self.cum_gemm = False

    def _gen_mat(self):
        mat = torch.zeros([int(torch.sum(self.kept2).item()), len(self.ranks)])
        ranges = []
        start_idx = 0
        while start_idx < len(self.ranks):
            # 当前元素
            current_value = self.ranks[start_idx]

            # 计算该元素的终止索引
            end_idx = start_idx
            while end_idx < len(self.ranks) and self.ranks[end_idx] == current_value:
                end_idx += 1

            # 保存该元素的起始和终止索引
            ranges.append([start_idx, end_idx])

            # 更新起始索引到下一个不同元素的位置
            start_idx = end_idx

        assert torch.sum(self.kept2) == len(ranges)
        for i, r in enumerate(ranges):
            mat[i][r[0] : r[1]] = 1

        return mat.permute(1, 0)

    def _create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = 384, 640
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = (
            torch.arange(*self.grid_conf["depth"], dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape
        xs = (
            torch.linspace(0, ogfW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, ogfH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return frustum

    def _get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )

        # cam_to_ego
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        Nprime = B * 80 * 24 * 40
        points = ((points - (self.bx - self.dx / 2.0)) / self.dx).long()
        points = points.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=points.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        points = torch.cat((points, batch_ix), 1)
        kept = (
            (points[:, 0] >= 0)
            & (points[:, 0] < self.nx[0])
            & (points[:, 1] >= 0)
            & (points[:, 1] < self.nx[1])
            & (points[:, 2] >= 0)
            & (points[:, 2] < self.nx[2])
        )
        points = points[kept]
        ranks = (
            points[:, 0] * (self.nx[1] * self.nx[2] * B)
            + points[:, 1] * (self.nx[2] * B)
            + points[:, 2] * B
            + points[:, 3]
        )
        sorts = ranks.argsort()

        points = points[sorts]
        ranks = ranks[sorts]

        kept2 = torch.ones(64095, device=points.device, dtype=torch.bool)
        kept2[:-1] = ranks[1:] != ranks[:-1]
        points = points[kept2]
        kept_t = torch.cat(
            (kept.nonzero(), torch.logical_not(kept).nonzero())
        ).squeeze()
        sorts_t = torch.cat((sorts, torch.arange(len(sorts), len(kept_t))))
        sorts_t = kept_t[sorts_t]
        return points, ranks, kept2, sorts_t, len(sorts)

    def _voxel_pooling(self, x):
        C, BN, *_ = x.shape
        x = torch.flatten(x, 1)
        x = x[:, self.sorts_t][:, : self.lens]
        # cumsum trick
        if self.cum_gemm:
            x = self._cumsum_trick_gemm(x)
        else:
            x = self._cumsum_trick_slice(x)
        # griddify (B x C x Z x X x Y)
        final = torch.zeros(
            (
                int(self.nx[0]),
                int(self.nx[1]),
                int(self.nx[2]),
                BN,
                C,
            ),
            device=x.device,
        )

        final[self.geom[:, 0], self.geom[:, 1], self.geom[:, 2], self.geom[:, 3], :] = x
        final = final.permute(3, 4, 2, 0, 1)[:, :, 0, :, :]
        return final

    def _cumsum_trick_gemm(self, x):
        x = x @ self.mat.to(x.device)
        return x.permute(1, 0)

    def _cumsum_trick_slice(self, x):
        x = x.cumsum(0)
        x = x[:, self.kept2]
        x = torch.cat((x[:1], x[1:] - x[:-1]))
        return x.permute(1, 0)

    def forward(self, feat: torch.Tensor, depth: torch.Tensor):
        """
        Args:
            feat: Extracted image features with shape (B, C, H, W)
            depth: Extracted depth features with shape (B, C, H, W)
        """

        # Expand to [B, C_d, C_i, H, W]
        x = depth[:, None, ...] * feat[:, :, None]
        x = x.permute(1, 0, 2, 3, 4)
        x = self._voxel_pooling(x)
        return x
