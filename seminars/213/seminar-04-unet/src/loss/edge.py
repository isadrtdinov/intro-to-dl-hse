import torch.nn as nn

from ..models import EdgeDetector
from .preprocess import PreprocessWrapper


class EdgeLoss(PreprocessWrapper):
    def __init__(self, unnorm=True, denoise=False):
        super().__init__(EdgeDetector(unnorm, denoise), nn.L1Loss())
