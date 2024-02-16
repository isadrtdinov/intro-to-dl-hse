from torch import nn


def init_weights(module):
    name = module.__class__.__name__
    if "Conv2d" in name or "ConvTranspose2d" in name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
