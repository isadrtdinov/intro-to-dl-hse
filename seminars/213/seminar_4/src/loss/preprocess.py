import torch.nn as nn
from torch import Tensor


class PreprocessWrapper(nn.Module):
    def __init__(self, preprocess, loss):
        super().__init__()
        self.preprocess = preprocess
        self.loss = loss

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.loss(self.preprocess(input), self.preprocess(target))
