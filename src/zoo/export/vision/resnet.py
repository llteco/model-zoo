import torch
import torchvision as tv
from torchvision.models.resnet import BasicBlock, Bottleneck

from .. import EXPORT


@EXPORT.register("resnet18")
class ResNet18(torch.nn.Module):
    compile_hier = [BasicBlock, Bottleneck]

    def __init__(self):
        super().__init__()
        self.model = tv.models.resnet18().eval()

    def forward(self, img):
        with torch.inference_mode():
            return self.model(img)

    @property
    def default_inputs(self):
        return {"img": torch.empty(1, 3, 224, 224)}

    @property
    def input_names(self):
        return ["img"]

    @property
    def output_names(self):
        return ["classify"]
