"""Simplified ReSplat encoder (inference/export only).

Derived from the upstream model; stripped of training-only logic (losses,
visualization dumps, AMP autocast, gradient checkpointing, multi-Gaussian
and local-kNN variants) and reduced to the DL3DV preset paths:
latent_downsample=4, fixed_latent_size, init_gaussian_multiple=1,
num_scales=1, no local knn, full-resolution depth prediction.

Module attribute names match the upstream encoder, so upstream checkpoints
(``encoder.*`` state dicts) load as-is.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from zoo.model.resplat.feature_wrapper import ResNetFeatureWarpper
from zoo.model.resplat.gaussians import build_covariance
from zoo.model.resplat.model_types import Gaussians
from zoo.model.resplat.point_transformer import (
    MultViewLowresAttn,
    PlainPointTransformer,
    PointLinearWrapper,
)
from zoo.model.resplat.projection import get_world_rays, sample_image_grid
from zoo.model.resplat.unimatch.mv_unimatch import MultiViewUniMatch

RGB2SH_C0 = 0.28209479177387814


def rgb2sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / RGB2SH_C0


@dataclass
class GaussianCfg:
    """Fields read by the (inlined) gaussian parameter decode."""

    sh_degree: int
    exp_scale_bias: float
    clamp_min_scale: float
    gaussian_scale_max: float

    @property
    def d_sh(self) -> int:
        return (self.sh_degree + 1) ** 2


@dataclass
class EncoderReSplatCfg:
    name: Literal["resplat"]
    num_depth_candidates: int
    gaussian_adapter: GaussianCfg
    downscale_factor: int
    shim_patch_size: int
    num_scales: int
    upsample_factor: int
    lowest_feature_resolution: int
    depth_unet_channels: int
    grid_sample_disable_cudnn: bool
    local_mv_match: int
    gaussian_regressor_channels: int
    monodepth_vit_type: str
    attn_proj_channels: int
    knn_samples: int
    num_blocks: int
    latent_downsample: int
    fixed_latent_size: bool
    num_refine: int
    num_basic_refine_blocks: int
    state_channels: int
    update_attn_proj_channels: int
    refine_knn_samples: int
    render_error_mv_attn_blocks: int


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class EncoderReSplat(nn.Module):
    def __init__(self, cfg: EncoderReSplatCfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.depth_predictor = MultiViewUniMatch(
            num_scales=cfg.num_scales,
            upsample_factor=cfg.upsample_factor,
            lowest_feature_resolution=cfg.lowest_feature_resolution,
            num_depth_candidates=cfg.num_depth_candidates,
            vit_type=cfg.monodepth_vit_type,
            unet_channels=cfg.depth_unet_channels,
            grid_sample_disable_cudnn=cfg.grid_sample_disable_cudnn,
            sample_log_depth=True,
            bilinear_upsample_depth=False,
            no_upsample_depth=False,
            use_amp=False,
            return_raw_mono_features=True,
        )

        # vitb mono features: 768ch at 1/16 -> pixel_shuffle(4) -> 192
        vit_channels = {"vits": 384, "vitb": 768, "vitl": 1024}[cfg.monodepth_vit_type]
        feature_upsampler_channels = vit_channels // 4 + 128 + 64 + 96 + 128

        self.gaussian_adapter_cfg = cfg.gaussian_adapter
        d_sh = cfg.gaussian_adapter.d_sh
        self.d_sh = d_sh

        # concat(img_unshuffle 48, latent_depth 1, match_prob 1, features)
        in_channels = 3 * 4**2 + 1 + 1 + feature_upsampler_channels
        channels = cfg.gaussian_regressor_channels
        self.gaussian_regressor = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

        # proj input: regressor channels + img_unshuffle + features + match prob
        self.proj = nn.Linear(
            channels + feature_upsampler_channels + 3 * 4**2 + 1, channels
        )

        # init point transformer over the latent point cloud
        self.pt = PlainPointTransformer(
            channels,
            cfg.knn_samples,
            num_blocks=cfg.num_blocks,
            attn_proj_channels=cfg.attn_proj_channels,
            with_mv_attn=True,
            with_mv_attn_lowres=True,
        )

        # gaussian head output: rotations 4 | scales 3 | opacity 1 |
        # sub-pixel offset 2 | sh 3*d_sh
        num_gaussian_parameters = 3 * d_sh + 4 + 3 + 2 + 1
        self.gaussian_head = nn.Sequential(
            nn.Linear(channels, num_gaussian_parameters),
            nn.GELU(),
            nn.Linear(num_gaussian_parameters, num_gaussian_parameters),
        )
        self.gaussian_head.apply(_init_weights)
        # upstream init: rotation rows random, all other rows zero
        nn.init.zeros_(self.gaussian_head[-1].weight[4:])
        nn.init.zeros_(self.gaussian_head[-1].bias[4:])

        if cfg.num_refine > 0:
            self.state_channels = cfg.state_channels
            self.update_proj = nn.Conv2d(channels, cfg.state_channels, 1)

            # refinement head output: means 3 | scales 3 | rotations 4 |
            # opacity 1 | sh  (the pixel-offset 2 channels are dropped)
            out_channels = 3 * d_sh + 4 + 3 + 1 + 3
            resnet_channels = 64 + 64 + 128  # ResNet-18 features

            self.update_rgb_error_proj = nn.Sequential(
                nn.Linear(3 * cfg.latent_downsample**2, resnet_channels),
                nn.LayerNorm(resnet_channels),
            )

            self.update_module = nn.Sequential(
                PointLinearWrapper(
                    out_channels + cfg.state_channels + resnet_channels,
                    cfg.state_channels,
                ),
                PlainPointTransformer(
                    cfg.state_channels,
                    cfg.refine_knn_samples,
                    num_blocks=cfg.num_basic_refine_blocks,
                    attn_proj_channels=cfg.update_attn_proj_channels
                    or cfg.attn_proj_channels,
                ),
            )

            self.update_head = nn.Sequential(
                nn.Linear(cfg.state_channels, cfg.state_channels),
                nn.GELU(),
                nn.Linear(cfg.state_channels, cfg.state_channels),
                nn.GELU(),
                nn.Linear(cfg.state_channels, cfg.state_channels),
                nn.GELU(),
                nn.Linear(cfg.state_channels, out_channels),
            )
            nn.init.zeros_(self.update_head[-1].weight)
            nn.init.zeros_(self.update_head[-1].bias)

            self.update_feature = ResNetFeatureWarpper(
                shallow_resnet_feature=False, resnet_layers=18
            )
            self.update_feature.eval()
            for params in self.update_feature.parameters():
                params.requires_grad = False

            self.update_error_attn = nn.ModuleList(
                [
                    MultViewLowresAttn(resnet_channels)
                    for _ in range(cfg.render_error_mv_attn_blocks)
                ]
            )

    def forward(self, context: dict) -> dict:
        """context: image [B,V,3,H,W], extrinsics [B,V,4,4] (c2w),
        intrinsics [B,V,3,3] (normalized), near/far [B,V].

        Returns {"gaussians": Gaussians, "condition_features": [BV,C,lh,lw]}.
        """
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        ld = self.cfg.latent_downsample

        # camera neighbour list for the depth predictor attention mask
        if v > 3:
            with torch.no_grad():
                xyzs = context["extrinsics"][:, :, :3, -1].detach()
                cameras_dist_index = torch.argsort(torch.cdist(xyzs, xyzs, p=2))
                cameras_dist_index = cameras_dist_index[
                    :, :, : (self.cfg.local_mv_match + 1)
                ]
        else:
            cameras_dist_index = None

        results_dict = self.depth_predictor(
            context["image"],
            attn_splits_list=[2],
            min_depth=1.0 / context["far"],
            max_depth=1.0 / context["near"],
            intrinsics=context["intrinsics"],
            extrinsics=context["extrinsics"],
            nn_matrix=cameras_dist_index,
        )
        depth = results_dict["depth_preds"][-1]  # [B, V, H, W]

        assert self.cfg.num_scales == 1

        # mono features [BV, 768, h/16, w/16] -> pixel_shuffle(4) -> 1/4
        mono_features = [
            F.interpolate(
                x, size=(h // 16, w // 16), mode="bilinear", align_corners=True
            )
            for x in results_dict["raw_mono_features"]
        ]
        mono_features = [F.pixel_shuffle(x, upscale_factor=4) for x in mono_features]
        mono_features = torch.cat(mono_features, dim=1)

        # cnn features (low->high res) -> all to latent resolution
        target_h, target_w = h // ld, w // ld
        cnn_features = results_dict["features_cnn_all_scales"][::-1]
        cnn_features = [
            F.interpolate(
                f, size=(target_h, target_w), mode="bilinear", align_corners=True
            )
            for f in cnn_features
        ]
        cnn_features = torch.cat(cnn_features, dim=1)

        mv_features = results_dict["features_mv"][0]
        if mv_features.shape[-2] != target_h or mv_features.shape[-1] != target_w:
            mv_features = F.interpolate(
                mv_features,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=True,
            )

        features = torch.cat((mono_features, cnn_features, mv_features), dim=1)

        # match probability at feature resolution -> latent
        match_prob = results_dict["match_probs"][-1]
        match_prob = torch.max(match_prob, dim=1, keepdim=True)[0]  # [BV, 1, H/8, W/8]

        # image unshuffle to the latent grid
        img_unshuffle = context["image"].reshape(b * v, 3, h, w)
        img_unshuffle = F.pixel_unshuffle(img_unshuffle, downscale_factor=4)

        latent_depth = F.interpolate(
            depth, size=(target_h, target_w), mode="bilinear", align_corners=True
        )

        if match_prob.shape[-2:] != latent_depth.shape[-2:]:
            match_prob = F.interpolate(
                match_prob, size=latent_depth.shape[-2:], mode="nearest"
            )

        concat = torch.cat(
            (
                img_unshuffle,
                latent_depth.reshape(b * v, 1, target_h, target_w),
                match_prob,
                features,
            ),
            dim=1,
        )

        out = self.gaussian_regressor(concat)

        # regressor output + all condition features (input to the refiner)
        out = torch.cat((out, img_unshuffle, features, match_prob), dim=1)

        lh, lw = latent_depth.shape[-2:]
        # 2D linear over pixels exactly like upstream (bv h w) c — a batched
        # 3D matmul would use different cuBLAS kernels (numeric drift)
        tmp_feature = self.proj(out.permute(0, 2, 3, 1).reshape(-1, out.shape[1]))

        # unproject latent depth into a point cloud
        xy_ray, _ = sample_image_grid((lh, lw), out.device)  # [lh, lw, 2]
        xy_ray = xy_ray.reshape(lh * lw, 1, 2)

        tmp_coords = xy_ray.unsqueeze(0).unsqueeze(0).repeat(b, v, 1, 1, 1)
        tmp_depth = latent_depth.reshape(b, v, lh * lw, 1, 1)
        tmp_extrinsics = context["extrinsics"].unsqueeze(2).unsqueeze(2)
        tmp_intrinsics = context["intrinsics"].unsqueeze(2).unsqueeze(2)
        origins, directions = get_world_rays(tmp_coords, tmp_extrinsics, tmp_intrinsics)
        point_cloud = origins + directions * tmp_depth  # [B, V, hw, 1, 3]

        offset = torch.tensor([v * lh * lw * i for i in range(1, b + 1)], device=device)
        point_cloud = point_cloud.reshape(b * v * lh * lw, 3)

        pt_output = self.pt((point_cloud, tmp_feature, offset), b=b, v=v, h=lh, w=lw)
        out = tmp_feature + pt_output

        condition_features = (
            out.reshape(b * v, lh, lw, -1).permute(0, 3, 1, 2).contiguous()
        )
        if self.cfg.num_refine == 0:
            condition_features = None

        out = self.gaussian_head(out)

        # ---- decode gaussian parameters (latent gaussians, repeat=1) ----
        gaussians = out.reshape(b * v, lh, lw, -1).permute(0, 3, 1, 2).float()
        gaussians = gaussians.reshape(b, v, -1, lh * lw)  # [B, V, C, hw]
        raw_gaussians = gaussians.permute(0, 1, 3, 2)  # [B, V, hw, C]

        num_sh = self.d_sh
        rotations_unnorm, scales, opacities_raw, offset_xy, sh = raw_gaussians.split(
            [4, 3, 1, 2, 3 * num_sh], dim=-1
        )
        opacities = opacities_raw.sigmoid()

        # sub-pixel ray offsets (+-0.5 px in latent space)
        offset_xy = offset_xy.sigmoid().unsqueeze(-2)  # [B, V, hw, 1, 2]
        pixel_size = 1 / torch.tensor((lw, lh), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

        # sh DC term from the input image at latent resolution
        sh_input_images = context["image"].reshape(b * v, 3, h, w)
        sh_input_images = F.interpolate(
            sh_input_images, size=(h // ld, w // ld), mode="area"
        )
        sh_input_images = sh_input_images.reshape(b, v, 3, -1).permute(0, 1, 3, 2)

        ac = self.gaussian_adapter_cfg
        scales = torch.clamp(
            F.softplus(scales - ac.exp_scale_bias),
            min=ac.clamp_min_scale,
            max=ac.gaussian_scale_max,
        )
        rotations = rotations_unnorm / (
            rotations_unnorm.norm(dim=-1, keepdim=True) + 1e-8
        )
        covariances = build_covariance(scales, rotations)
        c2w_rotations = context["extrinsics"][..., :3, :3].unsqueeze(
            2
        )  # [B, V, 1, 3, 3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        # means: unproject on the (sub-pixel shifted) rays
        origins, directions = get_world_rays(
            xy_ray,
            context["extrinsics"].unsqueeze(2).unsqueeze(2),
            context["intrinsics"].unsqueeze(2).unsqueeze(2),
        )
        depths = latent_depth.reshape(b, v, lh * lw, 1, 1)
        means = origins + directions * depths

        sh = sh.reshape(*sh.shape[:-1], 3, num_sh).clone()
        sh[..., 0] = sh[..., 0] + rgb2sh(sh_input_images)

        gaussians = Gaussians(
            means.reshape(b, -1, 3),
            covariances.reshape(b, -1, 3, 3),
            sh.reshape(b, -1, 3, num_sh),
            opacities.reshape(b, -1),
            scales=scales.reshape(b, -1, 3),
            rotations=rotations.reshape(b, -1, 4),
            rotations_unnorm=rotations_unnorm.reshape(b, -1, 4),
        )
        return {"gaussians": gaussians, "condition_features": condition_features}
