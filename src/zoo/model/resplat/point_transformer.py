import torch
import torch.nn as nn
import torch.nn.functional as F

from zoo.model.resplat.knn import KnnQuery, knn_query_and_group
from zoo.model.resplat.unimatch.dinov2.layers.block import Block as MultiViewBlock


class KNNAttention(nn.Module):
    def __init__(
        self,
        channels,
        knn_samples=16,
        num_heads=1,
        proj_channels=None,
    ):
        super().__init__()
        self.proj_channels = proj_channels

        self.knn_samples = knn_samples
        self.knn_query = KnnQuery(knn_samples)
        self.num_heads = num_heads
        assert self.num_heads == 1

        if self.proj_channels is not None:
            self.qkv = nn.Linear(channels, self.proj_channels * 3, bias=False)
            self.proj = nn.Linear(self.proj_channels, channels)
        else:
            self.qkv = nn.Linear(channels, channels * 3, bias=False)
            self.proj = nn.Linear(channels, channels)

    def forward(self, pxo, knn_idx=None):
        # [N, 3], [N, C], [B]
        p, x, o = pxo
        c = x.size(1)

        if self.proj_channels is not None:
            c = self.proj_channels

        assert c % self.num_heads == 0
        head_dim = c // self.num_heads
        scale_factor = head_dim**-0.5

        qkv = self.qkv(x)
        # [N, C]
        x_q, x_k, x_v = torch.chunk(qkv, chunks=3, dim=-1)

        # [N, K, C+3], [N, K]
        x_k, idx = knn_query_and_group(
            x_k.contiguous(),
            p,
            o,
            new_xyz=p,
            new_offset=o,
            idx=knn_idx,
            nsample=self.knn_samples,
            with_xyz=False,
        )

        # [N, K, C]
        x_v, _ = knn_query_and_group(
            x_v.contiguous(),
            p,
            o,
            new_xyz=p,
            new_offset=o,
            idx=idx,
            nsample=self.knn_samples,
            with_xyz=False,
        )

        n, k, c = x_k.shape

        # [N, 1, K]
        scores = torch.matmul(x_q.unsqueeze(1), x_k.permute(0, 2, 1)) * scale_factor
        # [N, C]
        out = torch.matmul(torch.softmax(scores, dim=2), x_v).squeeze(1)

        out = self.proj(out)

        return out


class MLP(nn.Module):
    def __init__(self, channels):
        super().__init__()

        expansion = 4

        self.fc1 = nn.Linear(channels, channels * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(channels * expansion, channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        channels,
        knn_samples=16,
        num_heads=1,
        attn_proj_channels=None,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = KNNAttention(
            channels,
            knn_samples=knn_samples,
            num_heads=num_heads,
            proj_channels=attn_proj_channels,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = MLP(channels)

    def forward(self, pxo, knn_idx=None):
        p, x, o = pxo

        x = x + self.attn((p, self.norm1(x), o), knn_idx=knn_idx)
        x = x + self.mlp(self.norm2(x))

        return x


class PlainPointTransformer(nn.Module):
    def __init__(
        self,
        channels,
        knn_samples=16,
        num_blocks=4,
        num_heads=1,
        attn_proj_channels=None,
        cache_knn_idx=True,
        with_mv_attn=False,
        with_mv_attn_lowres=False,
        mvattn_down_factor=4,
    ):
        super().__init__()

        self.cache_knn_idx = cache_knn_idx
        self.knn_samples = knn_samples
        self.knn_query = KnnQuery(knn_samples)

        self.with_mv_attn = with_mv_attn
        self.with_mv_attn_lowres = with_mv_attn_lowres

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                TransformerBlock(
                    channels,
                    knn_samples=knn_samples,
                    num_heads=num_heads,
                    attn_proj_channels=attn_proj_channels,
                )
            )

        # multi-view attention
        if self.with_mv_attn:
            self.mv_blocks = nn.ModuleList()
            for _ in range(num_blocks):
                if self.with_mv_attn_lowres:
                    self.mv_blocks.append(
                        MultViewLowresAttn(
                            channels,
                            down_factor=mvattn_down_factor,
                        )
                    )
                else:
                    self.mv_blocks.append(
                        MultiViewBlock(
                            channels,
                            num_heads=4,
                        )
                    )

    def compute_knn(
        self, p, o, b=None, v=None, h=None, w=None, extrinsics=None, intrinsics=None
    ):
        """Compute KNN indices for the given point cloud.

        Returns:
            knn_idx: [N, K] int tensor of KNN indices
        """
        return self.knn_query(p, p, o, o)

    def forward(
        self,
        pxo,
        b=None,
        v=None,
        h=None,
        w=None,
        extrinsics=None,
        intrinsics=None,
        cached_knn_idx=None,
        return_knn_idx=False,
    ):
        p, x, o = pxo
        # compute knn idx once and reuse across blocks (positions don't change)
        knn_idx = None
        if cached_knn_idx is not None:
            knn_idx = cached_knn_idx
        elif self.cache_knn_idx:
            knn_idx = self.compute_knn(
                p, o, b=b, v=v, h=h, w=w, extrinsics=extrinsics, intrinsics=intrinsics
            )

        num_blocks = len(self.mv_blocks) if self.with_mv_attn else len(self.blocks)

        if self.with_mv_attn:
            assert b is not None and v is not None and h is not None and w is not None
            for i in range(num_blocks):
                x = self.blocks[i]([p, x, o], knn_idx=knn_idx)
                # global multi-view attention
                x = x.reshape(b, v * h * w, x.shape[-1])
                if self.with_mv_attn_lowres:
                    x = self.mv_blocks[i](x, v=v, h=h, w=w)
                else:
                    x = self.mv_blocks[i](x)
                x = x.reshape(b * v * h * w, x.shape[-1])
        else:
            for blk in self.blocks:
                x = blk((p, x, o), knn_idx=knn_idx)

        if return_knn_idx:
            return x, knn_idx
        return x


class MultViewLowresAttn(nn.Module):
    def __init__(
        self,
        channels,
        down_factor=4,
        attn_proj_channels=None,
    ):
        super().__init__()

        self.down_factor = down_factor

        self.attn_proj_channels = attn_proj_channels

        if attn_proj_channels:
            ori_channels = channels
            self.proj0 = nn.Linear(channels, attn_proj_channels)
            channels = attn_proj_channels

        if self.down_factor == 8:
            down_factor = 4
        else:
            down_factor = self.down_factor

        self.proj1 = nn.Linear(channels * down_factor**2, channels)
        self.norm1 = nn.LayerNorm(channels)

        self.proj2 = nn.Linear(channels, channels * down_factor**2)
        self.norm2 = nn.LayerNorm(channels * down_factor**2)

        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

        if attn_proj_channels:
            self.proj3 = nn.Linear(channels, ori_channels)

        num_heads = 1 if self.attn_proj_channels else 4

        if channels % 32 != 0 and channels % 16 == 0:
            # flash attention 3 head_size should be a multiple of 8
            num_heads = 2

        if channels == 48 or channels == 3:
            # ablation: rgb error instead of feature error
            num_heads = 1

        self.attn = MultiViewBlock(channels, num_heads)

    def forward(self, x, v=None, h=None, w=None, y=None):
        if y is not None:
            return self.forward_cross_attn(x, y, v, h, w)
        residual = x
        if self.attn_proj_channels:
            x = self.proj0(x)

        # rearrange(x, "b (v h w) c -> (b v) c h w", v=v, h=h, w=w)
        x = (
            x.reshape(x.shape[0], v, h, w, x.shape[-1])
            .permute(0, 1, 4, 2, 3)
            .reshape(x.shape[0] * v, x.shape[-1], h, w)
        )

        if self.down_factor == 8:
            x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=True)
            down_factor = 4
        else:
            down_factor = self.down_factor

        x = F.pixel_unshuffle(x, down_factor)

        # rearrange(x, "(b v) c h w -> b (v h w) c", v=v)
        x = (
            x.reshape(-1, v, x.shape[1], x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
            .reshape(-1, v * x.shape[2] * x.shape[3], x.shape[1])
        )
        x = self.proj1(x)
        x = self.norm1(x)

        x = self.attn(x)

        x = self.proj2(x)
        x = self.norm2(x)

        # rearrange(x, "b (v h w) c -> (b v) c h w", v=v, h=h // df, w=w // df)
        h_low, w_low = h // self.down_factor, w // self.down_factor
        x = (
            x.reshape(x.shape[0], v, h_low, w_low, x.shape[-1])
            .permute(0, 1, 4, 2, 3)
            .reshape(x.shape[0] * v, x.shape[-1], h_low, w_low)
        )
        x = F.pixel_shuffle(x, down_factor)
        x = self.conv(x)
        if self.down_factor == 8:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        # rearrange(x, "(b v) c h w -> b (v h w) c", v=v)
        x = (
            x.reshape(-1, v, x.shape[1], x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
            .reshape(-1, v * x.shape[2] * x.shape[3], x.shape[1])
        )
        if self.attn_proj_channels:
            x = self.proj3(x)
        x = x + residual

        return x

    def forward_cross_attn(self, x, y, v=None, h=None, w=None):
        residual = x
        if self.attn_proj_channels:
            x = self.proj0(x)

        assert y is not None
        # rearrange(y, "b (v h w) c -> (b v) c h w", h=h, w=w)  # different v with x
        v_y = y.shape[1] // (h * w)
        y = (
            y.reshape(y.shape[0], v_y, h, w, y.shape[-1])
            .permute(0, 1, 4, 2, 3)
            .reshape(y.shape[0] * v_y, y.shape[-1], h, w)
        )
        num_cross_view = y.shape[0] // x.shape[0]

        # rearrange(x, "b (v h w) c -> (b v) c h w", v=v, h=h, w=w)
        x = (
            x.reshape(x.shape[0], v, h, w, x.shape[-1])
            .permute(0, 1, 4, 2, 3)
            .reshape(x.shape[0] * v, x.shape[-1], h, w)
        )

        if self.down_factor == 8:
            x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=True)
            y = F.interpolate(y, scale_factor=0.5, mode="bilinear", align_corners=True)
            down_factor = 4
        else:
            down_factor = self.down_factor

        x = F.pixel_unshuffle(x, down_factor)
        y = F.pixel_unshuffle(y, down_factor)

        # rearrange(x, "(b v) c h w -> b (v h w) c", v=v)
        x = (
            x.reshape(-1, v, x.shape[1], x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
            .reshape(-1, v * x.shape[2] * x.shape[3], x.shape[1])
        )
        # rearrange(y, "(b v) c h w -> b (v h w) c", v=num_cross_view)
        y = (
            y.reshape(-1, num_cross_view, y.shape[1], y.shape[2], y.shape[3])
            .permute(0, 1, 3, 4, 2)
            .reshape(-1, num_cross_view * y.shape[2] * y.shape[3], y.shape[1])
        )
        x = self.proj1(x)
        x = self.norm1(x)

        y = self.proj1(y)
        y = self.norm1(y)

        x = self.attn(x, y)

        x = self.proj2(x)
        x = self.norm2(x)

        # rearrange(x, "b (v h w) c -> (b v) c h w", v=v, h=h // df, w=w // df)
        h_low, w_low = h // self.down_factor, w // self.down_factor
        x = (
            x.reshape(x.shape[0], v, h_low, w_low, x.shape[-1])
            .permute(0, 1, 4, 2, 3)
            .reshape(x.shape[0] * v, x.shape[-1], h_low, w_low)
        )
        x = F.pixel_shuffle(x, down_factor)
        x = self.conv(x)
        if self.down_factor == 8:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        # rearrange(x, "(b v) c h w -> b (v h w) c", v=v)
        x = (
            x.reshape(-1, v, x.shape[1], x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
            .reshape(-1, v * x.shape[2] * x.shape[3], x.shape[1])
        )
        if self.attn_proj_channels:
            x = self.proj3(x)
        x = x + residual

        return x


class PointLinearWrapper(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, pxo, b=None, v=None, h=None, w=None):
        p, x, o = pxo
        x = self.linear(x)

        return [p, x, o]
