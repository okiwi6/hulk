from abc import ABC, abstractmethod

from timm.models._features import FeatureInfo
import torch
from torch import nn
import torch.nn.functional as F

from .blocks import C2f

class Neck(nn.Module, ABC):
    pass

class LastFeaturesOnlyNeck(Neck):
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return features[-1]


class ConcatFeaturesNeck(Neck):
    def __init__(self, feature_info: FeatureInfo):
        super().__init__()
        reduction = feature_info.reduction()
        scales = reduction[-1] / torch.tensor(reduction)
        self.pool_layers = nn.ModuleList(
            [nn.MaxPool2d(int(scale.item())) for scale in scales]
        )
        self.pointwise_conv = nn.Conv2d(
            sum(feature_info.channels()), feature_info.channels()[-1], kernel_size=1
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        concat_features = torch.cat(
            [
                pool_layer(feature)
                for pool_layer, feature in zip(self.pool_layers, features)
            ],
            dim=1,
        )
        return F.relu(self.pointwise_conv(concat_features))

class BiFpnNeck(Neck):
    def __init__(self, feature_info: FeatureInfo, n_last_channels: int, out_dim: int):
        super().__init__()
        self.n_last_channels = n_last_channels
        channels = feature_info.channels()[-n_last_channels:][::-1]
        self.conv_blocks = nn.ModuleList(
            [
                C2f(c1 + c2, c2, c2)
                for c1, c2 in zip(channels, channels[1:])
            ]
        )
        self.last_layer = C2f(channels[-1], out_dim, out_dim, stride=4)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        last_features = features[-self.n_last_channels:][::-1]
        x = last_features[0]
        for i, conv_block in enumerate(self.conv_blocks):
            x = torch.cat([F.upsample(x, scale_factor=2), last_features[i + 1]], dim=1)
            x = conv_block(x)
        return self.last_layer(x)
