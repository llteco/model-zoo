"""
Copyright (C) 2026 The MODEL-ZOO Authors.

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

import torchvision as tv

from zoo.model.matchformer.matchstereo import (
    DEFAULT_CHECKPOINT,
    GlobalCorrelation,
    MatchAttentionBlock,
    MatchAttentionLayer,
    MatchStereo,
    MetaFormer,
)

from .. import EXPORT, export_post_process


@EXPORT.register("matchstereo")
@export_post_process(
    [
        # TensorRT folds fp16 Div-by-subnormal-constant into Mul(1/c); 1/c
        # overflows fp16 and 0*inf emits NaN (all-NaN plain engines on TRT
        # 10.16/11.1). field_scale[..., 1] is a decayed dead weight
        # (~3.6e-07), so every layer tail hits this. See
        # xpu-low-level/docs/trt_spec/tensorrt_version_compat.md §6.
        "widen_subnormal_scaling",
    ]
)
class ExportMatchStereo(MatchStereo):
    """Exportable MatchStereo (tiny) with split MatchAttentionBlock.

    Each MatchAttentionLayer is decomposed into 8 extern-free glue blocks
    (phases between Linear/Conv boundaries).  compile_hier targets the glue
    types so hyperonnx captures each as a pure-triton kernel bundle.
    """

    hier = [MetaFormer, MatchAttentionBlock, GlobalCorrelation]
    # GlobalCorrelation is also captured as a triton bundle: TRT 10.3's fp16
    # fusion of its Sinkhorn tail emits NaN (see sub-module A/B in the 10.3
    # compatibility investigation), so dispatch it to the replay plugin there.
    compile_hier = [MatchAttentionLayer, GlobalCorrelation]
    fold_nodes_to_functions = False

    def __init__(
        self,
        refine_win_rs=[1, 1, 1, 1],
        refine_nums=[8, 8, 8, 2],
        num_heads=[4, 4, 4, 4],
        mlp_ratios=[2, 2, 2, 2],
        checkpoint=str(DEFAULT_CHECKPOINT),
    ):
        super().__init__(
            refine_win_rs=refine_win_rs,
            refine_nums=refine_nums,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            checkpoint=checkpoint,
        )

    @property
    def default_inputs(self):
        img0_file = DEFAULT_CHECKPOINT.parent / "im0_left.png"
        img1_file = DEFAULT_CHECKPOINT.parent / "im0_right.png"
        img0 = tv.io.decode_image(img0_file, mode=tv.io.ImageReadMode.RGB)[None]
        img1 = tv.io.decode_image(img1_file, mode=tv.io.ImageReadMode.RGB)[None]
        return {
            "img0": img0.float(),
            "img1": img1.float(),
        }

    @property
    def input_names(self):
        return ["img0", "img1"]

    @property
    def output_names(self):
        return ["field"]
