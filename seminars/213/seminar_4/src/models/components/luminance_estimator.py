import torch.nn as nn
from torch import Tensor


class LuminanceEstimator(nn.Module):
    def __init__(self, unnorm=True, renorm=False, keepshape=False):
        super().__init__()
        self.unnorm = unnorm
        self.renorm = renorm
        self.keepshape = keepshape

    def forward(self, x: Tensor) -> Tensor:
        channels = x.shape[1]
        if self.unnorm:
            x = x / 2 + 0.5
        # Power curve to linear RGB
        # x = x ** 2.2
        # Estimate luminance
        x = (0.2126 * x[:, 0] + 0.7152 * x[:, 1] + 0.0722 * x[:, 2]).unsqueeze(1)
        if self.renorm:
            # x = x ** (1 / 2.2) * 2 - 1
            x = x * 2 - 1
        if self.keepshape:
            x = x.repeat(1, channels, 1, 1)
        return x
