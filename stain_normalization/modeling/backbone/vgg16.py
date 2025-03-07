from torch import nn
import torchvision

def vgg16(weights: str | None = None) -> nn.Module:
    return torchvision.models.vgg16(weights=weights).features

