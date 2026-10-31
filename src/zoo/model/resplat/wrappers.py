"""Stage wrappers for ReSplat export.

The full two-stage inference is covered by two graphs:

  Stage0Wrapper      initial Gaussian prediction with the refinement init
                     folded in: images+poses -> Gaussians + recurrent
                     state + cached GT render-error features (the former
                     stage1_state_init / stage1_gtfeat graphs)
  Stage1StepWrapper  ONE refinement step with the rasterizer cut out:
                     rendered context images arrive as an input.

Stage1StateInit / Stage1GtFeatures are the torch references for the two
stage0 branches (used by the branch-level ONNX parity tests); they no
longer export as standalone graphs.
"""
# pylint: disable=unused-argument

import torch
import torch.nn.functional as F
from torch import nn

# ImageNet normalization used by the refinement error features
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """x: [N, 3, H, W] in [0, 1] -> ImageNet-normalized."""
    mean = x.new_tensor(_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def _init_state(update_proj, condition_features: torch.Tensor) -> torch.Tensor:
    """condition_features [BV, C, lh, lw] -> state [BV*lh*lw, C_state]."""
    state = update_proj(condition_features)  # [BV, C_state, lh, lw]
    bv, c, lh, lw = state.shape
    return state.permute(0, 2, 3, 1).reshape(bv * lh * lw, c)


def _gt_features(update_feature, image: torch.Tensor, latent_hw) -> torch.Tensor:
    """[B, V, 3, H, W] in [0, 1] -> input_view_features [BV, 256, lh, lw]."""
    b, v = image.shape[:2]
    x = image.reshape(b * v, *image.shape[2:])
    features = update_feature(_normalize(x))
    out = []
    for feat in features:
        if feat.shape[-2:] != latent_hw:
            feat = F.interpolate(
                feat, size=latent_hw, mode="bilinear", align_corners=True
            )
        out.append(feat)
    return torch.cat(out, dim=1)


class Stage0Wrapper(nn.Module):
    """Initial Gaussian prediction with the refinement initialization folded
    in. Must be built with num_refine > 0 (the update heads must exist);
    the intermediate condition_features never leave the graph."""

    def __init__(self, encoder) -> None:
        super().__init__()
        assert encoder.cfg.num_refine > 0, (
            "stage0 folds in the refinement init; build with num_refine > 0"
        )
        self.encoder = encoder
        self.update_proj = encoder.update_proj
        self.update_feature = encoder.update_feature

    @torch.no_grad()
    def forward(self, image, extrinsics, intrinsics, near, far):
        """Run stage0: encoder context -> (gaussians..., state, gt_features)."""
        context = {
            "image": image,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "near": near,
            "far": far,
        }
        out = self.encoder(context)
        g = out["gaussians"]
        state = _init_state(self.update_proj, out["condition_features"])
        gt_features = _gt_features(
            self.update_feature, image, (image.shape[-2] // 4, image.shape[-1] // 4)
        )
        return (
            g.means,
            g.covariances,
            g.harmonics,
            g.opacities,
            g.scales,
            g.rotations,
            g.rotations_unnorm,
            state,
            gt_features,
        )


class Stage1StateInit(nn.Module):
    """condition_features [BV, C_state, lh, lw] -> state [N, C_state].
    Torch reference for the state branch folded into Stage0Wrapper."""

    def __init__(self, encoder) -> None:
        super().__init__()
        self.update_proj = encoder.update_proj

    @torch.no_grad()
    def forward(self, condition_features):
        """Project condition features into the recurrent state."""
        return _init_state(self.update_proj, condition_features)


class Stage1GtFeatures(nn.Module):
    """GT context images -> input_view_features (cached across refine steps).
    Torch reference for the GT-feature branch folded into Stage0Wrapper."""

    def __init__(self, encoder, v: int, image_shape: tuple[int, int]) -> None:
        super().__init__()
        self.update_feature = encoder.update_feature
        self.v = v
        h, w = image_shape
        self.latent_hw = (h // 4, w // 4)

    @torch.no_grad()
    def forward(self, context_image):
        """Extract the cached GT render-error features."""
        return _gt_features(self.update_feature, context_image, self.latent_hw)


class Stage1StepWrapper(nn.Module):
    """One refinement step (rasterization external).

    Inputs:
      rendered_color        [1, V, 3, H, W]   external rasterizer output
      context_image         [1, V, 3, H, W]   GT context images
      input_view_features   [V, 256, lh, lw]  cached (stage0 GT features)
      state                 [N, C_state]      carried recurrent state
      prev_means            [1, N, 3]
      prev_scales           [1, N, 3]
      prev_rotations_unnorm [1, N, 4]
      prev_opacities_raw    [1, N, 1]         pre-sigmoid logits
      prev_shs              [1, N, 48]
      extrinsics_rot        [1, V, 3, 3]      context c2w rotations
      offset                [1] int64         pointops segment end-offset
    Outputs:
      state', means', scales', rotations_unnorm', rotations',
      opacities_raw', shs'
    """

    def __init__(self, encoder, v: int, image_shape: tuple[int, int]) -> None:
        super().__init__()
        self.update_feature = encoder.update_feature
        self.update_rgb_error_proj = encoder.update_rgb_error_proj
        self.update_error_attn = encoder.update_error_attn
        self.update_module = encoder.update_module
        self.update_head = encoder.update_head
        self.clamp_min_scale = encoder.cfg.gaussian_adapter.clamp_min_scale
        self.sh_dim = 3 * encoder.d_sh
        self.split_sizes = (3, 3, 4, 1, self.sh_dim)
        self.v = v
        h, w = image_shape
        self.latent_hw = (h // 4, w // 4)

    @torch.no_grad()
    def forward(
        self,
        rendered_color,
        context_image,
        input_view_features,
        state,
        prev_means,
        prev_scales,
        prev_rotations_unnorm,
        prev_opacities_raw,
        prev_shs,
        extrinsics_rot,  # graph-contract input (unused by the torch path)
        offset,
    ):
        """One refinement step on external render feedback -> updated tuple."""
        v = self.v
        b = 1
        lh, lw = self.latent_hw

        # ---- render error (feature branch) ----
        input0 = rendered_color.reshape(b * v, *rendered_color.shape[2:])
        features = self.update_feature(_normalize(input0))
        out = []
        for feat in features:
            if feat.shape[-2:] != self.latent_hw:
                feat = F.interpolate(
                    feat, size=self.latent_hw, mode="bilinear", align_corners=True
                )
            out.append(feat)
        render_view_features = torch.cat(out, dim=1)

        corr = render_view_features - input_view_features
        input_render_error = corr.permute(0, 2, 3, 1).reshape(
            b, v * lh * lw, corr.shape[1]
        )

        # ---- render error (RGB branch) ----
        rgb_render_error = rendered_color - context_image
        rgb_render_error = rgb_render_error.reshape(b * v, *rgb_render_error.shape[2:])
        rgb_render_error = F.pixel_unshuffle(rgb_render_error, 4)
        rgb_render_error = rgb_render_error.permute(0, 2, 3, 1).reshape(
            b, v * lh * lw, -1
        )
        input_render_error = input_render_error + self.update_rgb_error_proj(
            rgb_render_error
        )

        # ---- error attention ----
        for blk in self.update_error_attn:
            input_render_error = blk(input_render_error, v=v, h=lh, w=lw)

        # ---- recurrent update ----
        point_cloud = prev_means.reshape(-1, 3)
        tmp_gaussian = torch.cat(
            (
                prev_means,
                prev_scales,
                prev_rotations_unnorm,
                prev_opacities_raw,
                prev_shs,
            ),
            dim=-1,
        )
        tmp_gaussian = tmp_gaussian.reshape(-1, tmp_gaussian.shape[-1])
        tmp_render_error = input_render_error.reshape(-1, input_render_error.shape[-1])
        concat = torch.cat((tmp_gaussian, state, tmp_render_error), dim=-1)

        pxo = self.update_module[0]([point_cloud, concat, offset])
        state_out = self.update_module[1](pxo, b=b, v=v, h=lh, w=lw) + state

        delta_gaussians = self.update_head(state_out)
        delta_gaussians = delta_gaussians.reshape(b, -1, delta_gaussians.shape[-1])
        delta_means, delta_scales, delta_rotations, delta_opacities, delta_shs = (
            delta_gaussians.split(self.split_sizes, dim=-1)
        )

        # ---- parameter update ----
        means = prev_means + delta_means
        scales = (prev_scales + delta_scales).clamp(min=self.clamp_min_scale)
        rotations_unnorm = prev_rotations_unnorm + delta_rotations
        rotations = rotations_unnorm / (
            rotations_unnorm.norm(dim=-1, keepdim=True) + 1e-8
        )
        opacities_raw = prev_opacities_raw + delta_opacities
        shs = prev_shs + delta_shs

        return (
            state_out,
            means,
            scales,
            rotations_unnorm,
            rotations,
            opacities_raw,
            shs,
        )
