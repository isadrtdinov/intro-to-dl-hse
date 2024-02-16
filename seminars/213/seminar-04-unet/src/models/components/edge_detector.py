import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .gaussian_blur import GaussianBlur
from .luminance_estimator import LuminanceEstimator


class EdgeDetector(nn.Module):
    def __init__(self, unnorm = True, denoise = False):
        super().__init__()

        sobel_x = torch.tensor(
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]],
            dtype=torch.float32
        )
        sobel_x = sobel_x.repeat(1, 1, 1, 1)
        self.register_buffer("sobel_x", sobel_x, persistent=False)

        sobel_y = torch.transpose(sobel_x, 0, 1)
        sobel_y = sobel_y.repeat(1, 1, 1, 1)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

        self.gaussian = GaussianBlur(channels=3, kernel_size=3) if denoise \
            else nn.Identity()
        self.luma_estimator = LuminanceEstimator(unnorm)

    def forward(self, x: Tensor) -> Tensor:
        denoised = self.gaussian(x)
        luma = self.luma_estimator(denoised)
        edges = (F.conv2d(luma, self.sobel_x) ** 2 +
                 F.conv2d(luma, self.sobel_y) ** 2) ** (1 / 2)
        edges = F.pad(edges, (1, 1, 1, 1), mode="replicate")
        return edges
