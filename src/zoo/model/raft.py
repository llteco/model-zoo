#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Literal, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


def coords_corr(corr, idx, b, x, y):
    # corr: [N, H, W, H, W]
    H, W = corr.shape[-2:]
    mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    b = b.long()
    idx = idx.long()
    x = torch.clamp(x, 0, W - 1).long()
    y = torch.clamp(y, 0, H - 1).long()
    res = corr[b, idx[:, 2], idx[:, 1], y, x] * mask.float()
    print(mask.requires_grad, x.requires_grad, y.requires_grad, res.requires_grad)
    return res


def bilinear_sampling_corr(corr, idx1, idx2):
    """idx1: [M, (bhw)], idx2: [M, n_points, (bhw)]"""
    M, n_points = idx2.shape[:2]
    # reshape idx: [M * n_points, (bhw)]
    idx1 = idx1.unsqueeze(1).repeat(1, n_points, 1).view(-1, 3)
    idx2 = idx2.view(-1, 3)
    offset = idx2 - idx2.floor()
    dx, dy = offset[:, 1], offset[:, 2]
    b = idx2[:, 0].long()
    x0, y0 = idx2[:, 1].floor(), idx2[:, 2].floor()
    f00 = (1 - dy) * (1 - dx) * coords_corr(corr, idx1, b, x0, y0)
    res = f00
    return res.view(M, n_points)


def bilinear_sampler(img, coords, mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask_val = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask_val.float()

    return img


def coords_grid(batch, ht, wd, device, dtype):
    coords = torch.meshgrid(
        torch.arange(ht, device=device), torch.arange(wd, device=device)
    )
    coords = torch.stack(coords[::-1], dim=0).to(dtype)
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode="bilinear"):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class CorrBlock:
    def __init__(self, fmap1, fmap2, corr_levels, corr_radius):
        self.num_levels = corr_levels
        self.radius = corr_radius
        self.corr_pyramid = []
        # all pairs correlation
        for _ in range(self.num_levels):
            corr = CorrBlock.corr(fmap1, fmap2, 1)
            batch, h1, w1, dim, h2, w2 = corr.shape
            corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
            fmap2 = F.interpolate(
                fmap2, scale_factor=0.5, mode="bilinear", align_corners=False
            )
            self.corr_pyramid.append(corr)

    def __call__(self, coords, dilation=None):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        if dilation is None:
            dilation = torch.ones(
                batch, 1, h1, w1, device=coords.device, dtype=coords.dtype
            )

        # print(dilation.max(), dilation.mean(), dilation.min())
        out_pyramid = []
        for lvl in range(self.num_levels):
            corr = self.corr_pyramid[lvl]
            device = coords.device
            dx = torch.linspace(-r, r, 2 * r + 1, device=device, dtype=coords.dtype)
            dy = torch.linspace(-r, r, 2 * r + 1, device=device, dtype=coords.dtype)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), dim=-1)
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2).to(dtype=coords.dtype)
            delta_lvl = delta_lvl * dilation.view(batch * h1 * w1, 1, 1, 1)
            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**lvl
            coords_lvl = centroid_lvl + delta_lvl
            corr_sampled = bilinear_sampler(corr, coords_lvl)
            # corr_sampled might be a tuple if mask=True in bilinear_sampler
            if isinstance(corr_sampled, tuple):
                corr_sampled = corr_sampled[0]
            corr_sampled = corr_sampled.view(batch, h1, w1, -1)
            out_pyramid.append(corr_sampled)

        out = torch.cat(out_pyramid, dim=-1)
        out = out.permute(0, 3, 1, 2)
        return out

    @staticmethod
    def corr(fmap1, fmap2, num_head):
        batch, dim, h1, w1 = fmap1.shape
        h2, w2 = fmap2.shape[2:]
        fmap1 = fmap1.view(batch, num_head, dim // num_head, h1 * w1)
        fmap2 = fmap2.view(batch, num_head, dim // num_head, h2 * w2)
        corr = fmap1.transpose(2, 3) @ fmap2
        corr = corr.reshape(batch, num_head, h1, w1, h2, w2).permute(0, 2, 3, 1, 4, 5)
        return corr / torch.sqrt(torch.tensor(dim).float())


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()

        # self.sparse = sparse
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = norm_layer(planes)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.bn3 = norm_layer(planes)
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride), self.bn3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.relu(self.bn2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class LayerNorm(nn.Module):
    r"""
    LayerNorm that supports two data formats: channels_last (default)
    or channels_first. The ordering of the dimensions in the inputs.
    channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first
    corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNextBlock(nn.Module):
    r"""
    ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1
    Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) ->
    Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale.
            Default: 1e-6.
    """

    def __init__(self, dim, output_dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * output_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * output_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.final = nn.Conv2d(dim, output_dim, kernel_size=1, padding=0)

    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.final(input_x + x)
        return x


class ResNetFPN(nn.Module):
    """
    ResNet18, output resolution is 1/8.
    Each block has 2 layers.
    """

    def __init__(
        self,
        block_dims,
        initial_dim,
        pretrain,
        input_dim=3,
        output_dim=256,
        ratio=1.0,
        norm_layer=nn.BatchNorm2d,
        init_weight=False,
    ):
        super().__init__()
        # Config
        block = BasicBlock
        block_dims = [int(d * ratio) for d in block_dims]
        self.init_weight = init_weight
        self.input_dim = input_dim
        # Class Variable
        self.in_planes = initial_dim
        # Networks
        self.conv1 = nn.Conv2d(
            input_dim, initial_dim, kernel_size=7, stride=2, padding=3
        )
        self.bn1 = norm_layer(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        if pretrain == "resnet34":
            n_block = [3, 4, 6]
        elif pretrain == "resnet18":
            n_block = [2, 2, 2]
        else:
            raise NotImplementedError
        self.layer1 = self._make_layer(
            block, block_dims[0], stride=1, norm_layer=norm_layer, num=n_block[0]
        )  # 1/2
        self.layer2 = self._make_layer(
            block, block_dims[1], stride=2, norm_layer=norm_layer, num=n_block[1]
        )  # 1/4
        self.layer3 = self._make_layer(
            block, block_dims[2], stride=2, norm_layer=norm_layer, num=n_block[2]
        )  # 1/8
        self.final_conv = conv1x1(block_dims[2], output_dim)
        self._init_weights(pretrain)

    def _init_weights(self, pretrain):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.init_weight:
            from torchvision.models import (
                ResNet18_Weights,
                ResNet34_Weights,
                resnet18,
                resnet34,
            )

            if pretrain == "resnet18":
                pretrained_dict = resnet18(
                    weights=ResNet18_Weights.IMAGENET1K_V1
                ).state_dict()
            else:
                pretrained_dict = resnet34(
                    weights=ResNet34_Weights.IMAGENET1K_V1
                ).state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict
            }
            if self.input_dim == 6:
                for k, v in pretrained_dict.items():
                    if k == "conv1.weight":
                        pretrained_dict[k] = torch.cat((v, v), dim=1)
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)

    def _make_layer(self, block, dim, stride=1, norm_layer=nn.BatchNorm2d, num=2):
        layers = []
        layers.append(block(self.in_planes, dim, stride=stride, norm_layer=norm_layer))
        for _ in range(num - 1):
            layers.append(block(dim, dim, stride=1, norm_layer=norm_layer))
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x = self.relu(self.bn1(self.conv1(x)))
        for layer in self.layer1:
            x = layer(x)
        for layer in self.layer2:
            x = layer(x)
        for layer in self.layer3:
            x = layer(x)
        # Output
        output = self.final_conv(x)
        return output


class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_channel, dim=128):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = corr_channel
        self.convc1 = nn.Conv2d(cor_planes, dim * 2, 1, padding=0)
        self.convc2 = nn.Conv2d(dim * 2, dim + dim // 2, 3, padding=1)
        self.convf1 = nn.Conv2d(2, dim, 7, padding=3)
        self.convf2 = nn.Conv2d(dim, dim // 2, 3, padding=1)
        self.conv = nn.Conv2d(dim * 2, dim - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, corr_channel, num_blocks, hdim=128, cdim=128):
        # net: hdim, inp: cdim
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(corr_channel, dim=cdim)
        self.refine = []
        for _ in range(num_blocks):
            self.refine.append(ConvNextBlock(2 * cdim + hdim, hdim))
        self.refine = nn.ModuleList(self.refine)

    def forward(  # pylint: disable=unused-argument
        self, net, inp, corr, flow, upsample=True
    ):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        for blk in self.refine:
            net = blk(torch.cat([net, inp], dim=1))
        return net


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, mode="sintel"):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == "sintel":
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


class RAFT(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        radius: int = 4,
        iters: int = 4,
        block_dims: Sequence[int] = (64, 128, 256),
        initial_dim: int = 64,
        pretrain: Literal["resnet18", "resnet34"] = "resnet18",
    ):
        super().__init__()
        self.dim = dim
        self.iters = iters
        self.output_dim = dim * 2

        self.corr_levels = 4
        self.corr_radius = radius
        corr_channel = self.corr_levels * (radius * 2 + 1) ** 2

        self.cnet = ResNetFPN(
            block_dims=block_dims,
            initial_dim=initial_dim,
            pretrain=pretrain,
            input_dim=6,
            output_dim=2 * self.dim,
            norm_layer=nn.BatchNorm2d,
            init_weight=True,
        )

        # conv for iter 0 results
        self.init_conv = conv3x3(2 * dim, 2 * dim)
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(dim, dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, 64 * 9, 1, padding=0),
        )
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(dim, 2 * dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * dim, 6, 3, padding=1),
        )
        if iters > 0:
            self.fnet = ResNetFPN(
                block_dims=block_dims,
                initial_dim=initial_dim,
                pretrain=pretrain,
                input_dim=3,
                output_dim=self.output_dim,
                norm_layer=nn.BatchNorm2d,
                init_weight=True,
            )
            self.update_block = BasicUpdateBlock(
                corr_channel=corr_channel, num_blocks=4, hdim=dim, cdim=dim
            )

    def initialize_flow(self, img):
        """
        Flow is represented as difference between two coordinate grids.
        flow = coords2 - coords1
        """
        N, _, H, W = img.shape
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device, dtype=img.dtype)
        coords2 = coords_grid(N, H // 8, W // 8, device=img.device, dtype=img.dtype)
        return coords1, coords2

    def upsample_data(self, flow, info, mask):
        """Upsample [H/8, W/8, C] -> [H, W, C] using convex combination"""
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, (3, 3), padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, (3, 3), padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 8 * H, 8 * W), up_info.reshape(N, C, 8 * H, 8 * W)

    def forward(self, image1, image2):
        """Estimate optical flow between pair of frames"""
        N, _, H, W = image1.shape
        iters = self.iters

        flow_predictions = []
        info_predictions = []

        # padding
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        N, _, H, W = image1.shape
        dilation = torch.ones(
            N, 1, H // 8, W // 8, device=image1.device, dtype=image1.dtype
        )
        # run the context network
        cnet = self.cnet(torch.cat([image1, image2], dim=1))
        cnet = self.init_conv(cnet)
        net, context = torch.split(cnet, [self.dim, self.dim], dim=1)

        # init flow
        flow_update = self.flow_head(net)
        weight_update = 0.25 * self.upsample_weight(net)
        flow_8x = flow_update[:, :2]
        info_8x = flow_update[:, 2:]
        flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
        flow_predictions.append(flow_up)
        info_predictions.append(info_up)

        if self.iters > 0:
            # run the feature network
            fmap1_8x = self.fnet(image1)
            fmap2_8x = self.fnet(image2)
            corr_fn = CorrBlock(fmap1_8x, fmap2_8x, self.corr_levels, self.corr_radius)
        else:
            corr_fn = None

        for _ in range(iters):
            if corr_fn is None:
                break
            N, _, H, W = flow_8x.shape
            flow_8x = flow_8x.detach()
            coords = coords_grid(N, H, W, device=image1.device, dtype=image1.dtype)
            coords2 = (coords + flow_8x).detach()
            corr = corr_fn(coords2, dilation=dilation)
            net = self.update_block(net, context, corr, flow_8x)
            flow_update = self.flow_head(net)
            weight_update = 0.25 * self.upsample_weight(net)
            flow_8x = flow_8x + flow_update[:, :2]
            info_8x = flow_update[:, 2:]
            # upsample predictions
            flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        for i, _ in enumerate(info_predictions):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])

        return {
            "final": flow_predictions[-1],
            "flow": flow_predictions,
            "info": info_predictions,
        }
