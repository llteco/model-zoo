from pathlib import Path

import torch
import torchvision as tv
from torchvision.models.resnet import BasicBlock, Bottleneck

from .. import EXPORT

_MODELS = Path(__file__).parents[4] / "models"
_WEIGHTS = tv.models.ResNet18_Weights.IMAGENET1K_V1
# ImageNet-1k normalization (what _WEIGHTS.transforms() applies).
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


@EXPORT.register("resnet18")
class ResNet18(torch.nn.Module):
    compile_hier = [BasicBlock, Bottleneck]

    def __init__(self):
        super().__init__()
        self.model = tv.models.resnet18(weights=_WEIGHTS).eval()

    def forward(self, img):
        with torch.inference_mode():
            return self.model(img)

    @property
    def default_inputs(self):
        # ImageNet validation photo stored pre-resized to 224x224: only
        # normalize (no resize/center-crop) so the subject stays in frame.
        img = tv.io.decode_image(
            str(_MODELS / "vision/siamese.jpg"),
            mode=tv.io.ImageReadMode.RGB,
        )
        img = img.to(torch.float32).div(255.0)
        img = tv.transforms.Normalize(_MEAN, _STD)(img)
        return {"img": img[None]}

    @property
    def input_names(self):
        return ["img"]

    @property
    def output_names(self):
        return ["classify"]
