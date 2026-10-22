"""Config preset + encoder builder for the simplified ReSplat model.

Values transcribed from the upstream hydra configs
(config/model/encoder/resplat.yaml overridden by config/experiment/dl3dv.yaml)
for the DL3DV 8-view 512x960 preset.
"""

from zoo.model.resplat.encoder_resplat import EncoderReSplatCfg, GaussianCfg


def dl3dv_cfg(num_refine: int = 1) -> EncoderReSplatCfg:
    """DL3DV preset (view8 512x960 checkpoint family). Set num_refine >= 1 to
    construct refinement modules."""
    return EncoderReSplatCfg(
        name="resplat",
        num_depth_candidates=128,
        gaussian_adapter=GaussianCfg(
            sh_degree=3,
            exp_scale_bias=4.0,
            clamp_min_scale=1e-6,
            gaussian_scale_max=1.0,
        ),
        downscale_factor=4,
        shim_patch_size=16,
        num_scales=1,
        upsample_factor=8,
        lowest_feature_resolution=8,
        depth_unet_channels=128,
        grid_sample_disable_cudnn=False,
        local_mv_match=2,
        gaussian_regressor_channels=512,
        monodepth_vit_type="vitb",
        attn_proj_channels=64,
        knn_samples=16,
        num_blocks=6,
        latent_downsample=4,
        fixed_latent_size=True,
        num_refine=num_refine,
        num_basic_refine_blocks=4,
        state_channels=512,
        update_attn_proj_channels=64,
        refine_knn_samples=8,
        render_error_mv_attn_blocks=1,
    )


# fixed seed so every export/benchmark process bakes in the SAME random
# weights as the parity baseline (tests/resplat_parity_baseline.pt)
REPRO_SEED = 1234


def build_encoder(num_refine: int = 1, checkpoint: str | None = None):
    """Build the encoder. Without a checkpoint the weights are random
    (ImageNet ResNet-18 for the frozen render-error extractor, seeded with
    REPRO_SEED for reproducible exports); with one, ``encoder.*``-prefixed
    keys are loaded non-strictly."""
    import torch

    from zoo.model.resplat.encoder_resplat import EncoderReSplat

    torch.manual_seed(REPRO_SEED)
    encoder = EncoderReSplat(dl3dv_cfg(num_refine=num_refine))
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        ckpt = {
            k[len("encoder.") :]: v for k, v in ckpt.items() if k.startswith("encoder.")
        }
        missing, unexpected = encoder.load_state_dict(ckpt, strict=False)
        print(
            f"loaded checkpoint: {len(ckpt)} keys, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return encoder.eval()
