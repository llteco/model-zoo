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

import torch

from zoo.model.matchformer.matchstereo import (
    AttentionBlock,
    GlobalCorrelation,
    MatchAttentionBlock,
    MatchStereo,
    MetaFormer,
)

from .. import EXPORT


@EXPORT.register("matchstereo")
class ExportMatchStereo(MatchStereo):
    """Exportable MatchStereo (tiny) with pad-to-32 and compile hierarchy."""

    # profiled @384x640 fp32 (eager -> compiled ms): MetaFormer 3.1->1.6,
    # AttentionBlock 1.3->0.8, GlobalCorrelation 1.4->0.2, MatchAttentionBlock
    # 13.7->3.7; UpConv regresses on 2/3 instances (0.18->0.41, 0.16->0.20),
    # stays plain ONNX
    compile_hier = [MetaFormer, AttentionBlock, GlobalCorrelation, MatchAttentionBlock]
    fold_nodes_to_functions = False

    @property
    def default_inputs(self):
        return {
            "img0": torch.empty(1, 3, 384, 640),
            "img1": torch.empty(1, 3, 384, 640),
        }

    @property
    def input_names(self):
        return ["img0", "img1"]

    @property
    def output_names(self):
        return ["field"]
