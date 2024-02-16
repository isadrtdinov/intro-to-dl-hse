import torch
import torchvision

from torch import nn
from torch import Tensor
from typing import Tuple


def norm(x: Tensor, mean: Tuple, std: Tuple):
    return torch.stack([(x[:, i] - mean[i]) / std[i] for i in range(x.shape[1])], dim=1)


class VGGPerceptualLoss(nn.Module):
    def __init__(self, avgpool=False, unnorm=True):
        super().__init__()
        self.unnorm = unnorm

        # Get pretrained VGG model
        self.vgg_model = torchvision.models.vgg16(pretrained=True)
        # Remove classifier part
        self.vgg_model.classifier = nn.Identity()
        if not avgpool:
            self.vgg_model.avgpool = nn.Identity()
        # Remove layers with deep features
        self.vgg_model.features = nn.Sequential(*self.vgg_model.features[:22])
        # Freeze model
        self.vgg_model.eval()
        for param in self.vgg_model.parameters():
            param.requires_grad = False
        # L1 loss instance
        self.loss = nn.L1Loss()

        self.mean = (0.48235, 0.45882, 0.40784)
        self.std = (0.229, 0.224, 0.225)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.unnorm:
            input = input / 2 + 0.5
            target = target / 2 + 0.5
        out = self.loss(self.vgg_model(norm(input, self.mean, self.std)),
                        self.vgg_model(norm(target, self.mean, self.std)))
        return out
