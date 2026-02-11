"""
Copyright (C) 2026 The ZOO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import enum
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class Layout(enum.Enum):
    """Possible Bayer color filter array layouts.

    The value of each entry is the color index (R=0,G=1,B=2)
    within a 2x2 Bayer block.
    """

    RGGB = (0, 1, 1, 2)
    GRBG = (1, 0, 2, 1)
    GBRG = (1, 2, 0, 1)
    BGGR = (2, 1, 1, 0)


def _ranged_tanh(min_value: float = 0.5, max_value: float = 2.0):
    def get_activation(left, right):
        def activation(x):
            return (torch.tanh(x) * 0.5 + 0.5) * (right - left) + left

        return activation

    return get_activation(min_value, max_value)


class Guide(nn.Module):
    """Network to generate a guidance map

    Input is a full size RGB image, output a 12-channel full size guidance map.
    """

    def __init__(self, in_channels=4):
        super().__init__()
        self.channel_mixing = nn.Conv2d(in_channels, 1, 1)
        self.tanh = nn.Tanh()

    def gen_grid(self, shape):
        bs, H, W, _ = shape
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(-H, H, H),
            torch.linspace(-W, W, W),
        )
        ref_y = ref_y[None] / H
        ref_x = ref_x[None] / W

        ref_2d = torch.stack((ref_x, ref_y), -1).cuda()
        ref_2d = ref_2d.repeat(bs, 1, 1, 1)
        return ref_2d

    def forward(self, inputs):
        guidemap = self.channel_mixing(inputs).permute(0, 2, 3, 1).contiguous()
        guidemap = self.tanh(guidemap).contiguous()
        ref = self.gen_grid(guidemap.shape).to(dtype=inputs.dtype, device=inputs.device)
        guidemap = torch.cat((ref, guidemap), -1)
        return guidemap


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize=3, stride=1, bias=True):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            padding_mode="zeros",
            bias=bias,
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))


class Debayer3x3(torch.nn.Module):
    """Demosaicing of Bayer images using 3x3 convolutions.

    Compared to Debayer2x2 this method does not use upsampling.
    Instead, we identify five 3x3 interpolation kernels that
    are sufficient to reconstruct every color channel at every
    pixel location.

    We convolve the image with these 5 kernels using stride=1
    and a one pixel reflection padding. Finally, we gather
    the correct channel values for each pixel location. Todo so,
    we recognize that the Bayer pattern repeats horizontally and
    vertically every 2 pixels. Therefore, we define the correct
    index lookups for a 2x2 grid cell and then repeat to image
    dimensions.
    """

    def __init__(self, layout: Layout = Layout.GRBG):
        super(Debayer3x3, self).__init__()
        self.layout = layout
        # fmt: off
        self.kernels = torch.nn.Parameter(
            torch.tensor(
                [
                    [0, 0.25, 0],
                    [0.25, 0, 0.25],
                    [0, 0.25, 0],

                    [0.25, 0, 0.25],
                    [0, 0, 0],
                    [0.25, 0, 0.25],

                    [0, 0, 0],
                    [0.5, 0, 0.5],
                    [0, 0, 0],

                    [0, 0.5, 0],
                    [0, 0, 0],
                    [0, 0.5, 0],
                ]
            ).view(4, 1, 3, 3),
            requires_grad=False,
        )
        # fmt: on

        self.index = torch.nn.Parameter(
            self._index_from_layout(layout),
            requires_grad=False,
        )

    def forward(self, x):
        """Debayer image.

        Parameters
        ----------
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
            Color images in RGB channel order.
        """
        B, C, H, W = x.shape

        xpad = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="reflect")
        c = torch.nn.functional.conv2d(xpad, self.kernels, stride=1)
        c = torch.cat((c, x), 1)  # Concat with input to give identity kernel Bx5xHxW

        rgb = torch.gather(
            c,
            1,
            self.index.repeat(
                1,
                1,
                torch.div(H, 2, rounding_mode="floor"),
                torch.div(W, 2, rounding_mode="floor"),
            ).expand(
                B, -1, -1, -1
            ),  # expand in batch is faster than repeat
        )

        return rgb

    def _index_from_layout(self, layout: Layout) -> torch.Tensor:
        """Returns a 1x3x2x2 index tensor for each color RGB in a 2x2 bayer tile.

        Note, the index corresponding to the identity kernel is 4, which will be
        correct after concatenating the convolved output with the input image.
        """
        #       ...
        # ... b g b g ...
        # ... g R G r ...
        # ... b G B g ...
        # ... g r g r ...
        #       ...
        # fmt: off
        rggb = torch.tensor(
            [
                # dest channel r
                [4, 2],  # pixel is R,G1
                [3, 1],  # pixel is G2,B
                # dest channel g
                [0, 4],  # pixel is R,G1
                [4, 0],  # pixel is G2,B
                # dest channel b
                [1, 3],  # pixel is R,G1
                [2, 4],  # pixel is G2,B
            ]
        ).view(1, 3, 2, 2)
        # fmt: on
        return {
            Layout.RGGB: rggb,
            Layout.GRBG: torch.roll(rggb, 1, -1),
            Layout.GBRG: torch.roll(rggb, 1, -2),
            Layout.BGGR: torch.roll(rggb, (1, 1), (-1, -2)),
        }[layout]


class AITM3(nn.Module):
    """Model of AI-ISP tone-mapping V3."""

    def __init__(
        self,
        gamma_range: tuple[float, float, float, float] = (1, 2.5, 0, 1),
        layout: Literal["rggb", "bggr", "grbg", "gbrg"] = "grbg",
    ):
        super().__init__()
        if layout == "rggb":
            self.demosaic = Debayer3x3(Layout.RGGB)
        elif layout == "bggr":
            self.demosaic = Debayer3x3(Layout.BGGR)
        elif layout == "grbg":
            self.demosaic = Debayer3x3(Layout.GRBG)
        elif layout == "gbrg":
            self.demosaic = Debayer3x3(Layout.GBRG)
        else:
            raise ValueError(f"Unsupported layout: {layout}")
        self.gamma_range = gamma_range
        assert len(gamma_range) == 4, "gamma_range should be a tuple of 4 floats"

        self.guide = Guide()
        self.fold = nn.PixelUnshuffle(2)
        self.unfold = nn.PixelShuffle(2)

        self.head1 = BaseConv(4, 8, ksize=3, stride=2)
        self.body1 = BaseConv(8, 16, ksize=3, stride=2)

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.image_adaptive_global = nn.Sequential(
            nn.Linear(16, 32, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(32, 2, bias=False),
        )

        self.image_adaptive_local = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=1, padding=1, padding_mode="zeros", bias=True),
            nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=1, padding=1, padding_mode="zeros", bias=True),
        )

    def apply_gamma(self, img, gamma, threshold):
        mean_val = torch.mean(img, dim=1, keepdim=True)
        gain = torch.where(
            mean_val < threshold,
            threshold ** (1 / gamma - 1 - 1e-5),
            mean_val ** (1 / gamma) / (mean_val + 1e-5),
        )

        out = img * gain

        return out

    def forward(self, bayer):
        b, c, h, w = bayer.shape
        fold = self.fold(bayer)
        guidemap = self.guide(fold).unsqueeze(1)  # B, 1, H, W, C
        down = F.interpolate(fold, size=(256, 256), mode="bilinear")

        fea = self.head1(down)
        fea = self.body1(fea)

        fea_global = self.pooling(fea)
        fea_global = fea_global.view(fea_global.shape[0], fea_global.shape[1])
        param_global = self.image_adaptive_global(fea_global)

        param_local = self.image_adaptive_local(fea)
        _, _, m, n = param_local.shape
        param_local = param_local.view(b, 3, 8, m, n)

        b, c, d, m, n = param_local.shape
        param_local_2d = param_local.view(b, c * d, m, n)
        grid_xy = guidemap.squeeze(1)[..., :2]
        grid_z = guidemap.squeeze(1)[..., 2:]

        coeffmap = torch.nn.functional.grid_sample(
            param_local_2d, grid_xy, align_corners=False
        )
        _, _, H, W = coeffmap.shape
        coeffmap = coeffmap.view(b, c, d, H, W)

        z = (grid_z + 1) * d / 2 - 0.5
        z = z.permute(0, 3, 1, 2)

        z0 = torch.floor(z).long()
        z1 = z0 + 1

        w1 = z - z0.to(z.dtype)
        w0 = 1 - w1

        z0_clamp = z0.clamp(0, d - 1)
        z1_clamp = z1.clamp(0, d - 1)

        idx0 = z0_clamp.unsqueeze(1).expand(-1, c, -1, -1, -1)
        idx1 = z1_clamp.unsqueeze(1).expand(-1, c, -1, -1, -1)

        val0 = torch.gather(coeffmap, 2, idx0).squeeze(2)
        val1 = torch.gather(coeffmap, 2, idx1).squeeze(2)

        mask0 = (z0 >= 0) & (z0 < d)
        mask1 = (z1 >= 0) & (z1 < d)

        val0 = val0 * mask0
        val1 = val1 * mask1

        coeffmap = val0 * w0 + val1 * w1

        param_gamma = _ranged_tanh(*self.gamma_range[:2])(param_global[:, 0:1])[
            ..., None, None
        ]
        param_threshold = _ranged_tanh(*self.gamma_range[2:4])(param_global[:, 1:2])[
            ..., None, None
        ]

        out_global = self.apply_gamma(fold, param_gamma, param_threshold)

        local_map = torch.mean(
            coeffmap.clamp(
                0,
            )
            + 1,
            dim=1,
            keepdim=True,
        )
        out = (fold * local_map * out_global + out_global) / 2

        unfold_out = self.unfold(out)

        rgb_out = self.demosaic(unfold_out).clip(1e-5, 1)
        return unfold_out ** (1 / 2.2), rgb_out ** (1 / 2.2)
