import torch
import torch.nn.functional as F
from torch import nn, Tensor


class GaussianBlur(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.pad = [kernel_size // 2] * 4

        mean = kernel_size // 2
        sigma = 0.3 * (mean - 1) + 0.8

        # Create x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size, 1)
        y_grid = x_grid.T
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        variance = sigma ** 2.0
        gaussian_kernel = torch.exp(
            -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
        )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        self.register_buffer("kernel", gaussian_kernel, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, self.pad, mode="replicate")
        return F.conv2d(x, self.kernel, groups=self.channels)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.channels}, kernel_size={self.kernel_size})"
