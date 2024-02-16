import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def random_shift_bounds(left_amount, right_amount):
    shift = np.random.randint(-left_amount, right_amount + 1)
    a_min = left_amount + shift
    a_max = -right_amount + shift
    if a_max == 0:
        return slice(a_min, None)
    return slice(a_min, a_max)


class RandomShift(nn.Module):
    def __init__(self, distance = (0, 1, 0, 1)):
        super().__init__()
        self.distance = distance  # Left right top bottom

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            x = F.pad(x, self.distance, mode="replicate")
            x = x[:, :, random_shift_bounds(*self.distance[:2]), :]  # Vertical shift
            x = x[:, :, :, random_shift_bounds(*self.distance[2:])]  # Horizontal shift
        return x
