from torch import nn, Tensor

from .components import RandomShift, GaussianBlur


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, num_feature_levels, hidden_channels,
                 activation: nn.Module = nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()

        features = [nn.Conv2d(in_channels, hidden_channels, kernel_size=4,
                              stride=2, padding=1, bias=False)]

        channels = [hidden_channels * 2 ** c
                    for c in range(num_feature_levels)]
        channels[-1] = out_channels
        for i in range(num_feature_levels - 1):
            features += [activation,
                         nn.Conv2d(channels[i], channels[i + 1], kernel_size=4,
                                   stride=2, padding=1, bias=False)]
            if i < num_feature_levels - 2:
                features += [nn.BatchNorm2d(channels[i + 1])]

        self.features = nn.Sequential(*features)
        self.preprocess = nn.Sequential(
            RandomShift(),
            GaussianBlur(channels=in_channels, kernel_size=5)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.preprocess(x)
        return self.features(x)

    def requires_grad(self, requires_grad: bool = True):
        if requires_grad:
            self.train()
            for param in self.parameters():
                param.requires_grad = True
        else:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
