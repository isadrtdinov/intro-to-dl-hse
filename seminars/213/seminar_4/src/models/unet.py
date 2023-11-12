import torch
from torch import nn, Tensor

# Template taken from week06_vision small hw


class ConvBlock(nn.Module):
    """
       Convolutional Block, includes several sequential convolutional and activation layers.
       Hint: include BatchNorm here
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias = True,
                 block_depth = 1, activation = nn.ReLU(inplace=True),
                 norm = None, dropout = 0):
        """
            in_channels - input channels num
            out_channels - output channels num
            kernel_size - kernel size for convolution layers
            stride - convolution stride
            bias - use bias in convolution
            block_depth - number of convolution + activation repetitions
            activation - activation function instance
            norm - normalization layer (optional)
            dropout - dropout probability
        """
        super().__init__()

        # your code
        features = [activation,
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=(kernel_size - 1) // 2, bias=bias)]
        if norm is not None:
            features += [norm(out_channels)]
        if dropout:
            features += [nn.Dropout(p=dropout)]
        for _ in range(block_depth - 1):
            features += [activation,
                         nn.Conv2d(out_channels, out_channels, kernel_size,
                                   padding=kernel_size // 2, bias=bias)]
            if norm is not None:
                features += [norm(out_channels)]
            if dropout:
                features += [nn.Dropout(p=dropout)]

        self.features = nn.Sequential(*features)

    def forward(self, x):
        x = self.features(x)
        return x


class DeconvBlock(nn.Module):
    """
        Decoding block, includes several sequential convolutional and activation layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias = True,
                 block_depth = 1, activation = nn.ReLU(inplace=True),
                 post_activation = None, norm = None, dropout: float = 0):
        """
            in_channels - input channels num
            out_channels - output channels num
            kernel_size - kernel size for convolution layers
            stride - convolution stride
            bias - use bias in convolution
            block_depth - number of convolution + activation repetitions
            activation - activation function instance
            norm - normalization layer (optional)
            dropout - dropout probability
        """
        super().__init__()

        # your code
        features = [activation,
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                       padding=(kernel_size - 1) // 2, bias=bias)]
        if norm is not None:
            features += [norm(out_channels)]
        if dropout:
            features += [nn.Dropout(p=dropout)]
        for _ in range(block_depth - 1):
            features += [activation,
                         nn.Conv2d(out_channels, out_channels, kernel_size,
                                   padding=kernel_size // 2, bias=bias)]
            if norm is not None:
                features += [norm(out_channels)]
            if dropout:
                features += [nn.Dropout(p=dropout)]

        if post_activation is not None:
            features += [post_activation]

        self.features = nn.Sequential(*features)

    def forward(self, x):
        x = self.features(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_feature_levels,
                 hidden_channels, max_hidden_channels, block_depth = 1,
                 dropout_layers = [0, 0.5, 0.5, 0.5]):
        """
        Input:
            in_channels - input number of channels (1 for gray images, 3 for rgb)
            out_channels - output number of channels
            feature_levels_num - number of down- and up- block levels
            hidden_channels - output number of channels of the first Convolutional Block
            max_hidden_channels - maximum number of channels in hidden layers
            block_depth - number of convolutions + activations in one Convolutional Block
            dropout_layers - dropout probablilities for upsampling layers
        """
        super().__init__()
        self.dropout_layers = dropout_layers + [0] * (num_feature_levels - len(dropout_layers))

        down_blocks = [nn.Conv2d(in_channels, hidden_channels,
                                 kernel_size=4, stride=2, padding=1, bias=False)]
        up_blocks = []

        up_channels = [min(hidden_channels * 2 ** c, max_hidden_channels)
                       for c in range(num_feature_levels)]
        down_channels = list(reversed(up_channels))
        down_channels[0] //= 2
        for i in range(num_feature_levels - 1):
            # your code
            # fill self.down_blocks and self.up_blocks with DownBlock/UpBlock
            # each DownBlock/UpBlock increase/decrease number of channels by 2 times
            down_blocks += [ConvBlock(up_channels[i], up_channels[i + 1], kernel_size=4,
                                      stride=2, bias=False, block_depth=block_depth,
                                      activation=nn.LeakyReLU(0.2, inplace=True),
                                      norm=nn.BatchNorm2d if i < num_feature_levels - 2
                                                          else None)]

        for i in range(num_feature_levels - 1):
            up_blocks += [DeconvBlock(down_channels[i] * 2, down_channels[i + 1], kernel_size=4,
                                      stride=2, block_depth=block_depth, bias=False,
                                      activation=nn.ReLU(inplace=True),
                                      norm=nn.BatchNorm2d, dropout=self.dropout_layers[i])]
        up_blocks += [DeconvBlock(down_channels[-1] * 2, out_channels, kernel_size=4,
                                  stride=2, bias=True, block_depth=block_depth,
                                  activation=nn.ReLU(inplace=True), post_activation=nn.Tanh(),
                                  norm=None, dropout=self.dropout_layers[-1])]

        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)

    def forward(self, x: Tensor) -> Tensor:
        activations = []
        for down_block in self.down_blocks:
            x = down_block(x)
            activations += [x]
        x = torch.tensor([], device=x.device)
        for up_block in self.up_blocks:
            x = up_block(torch.cat([x, activations.pop()], dim=1))

        return x
