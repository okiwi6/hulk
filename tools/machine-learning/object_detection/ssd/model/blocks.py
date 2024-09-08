import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, activation: bool, stride: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)


class C2f(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, hidden_channels: int, stride: int = 1
    ):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, hidden_channels, True)
        self.conv2 = ConvBlock(hidden_channels, out_channels, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x
