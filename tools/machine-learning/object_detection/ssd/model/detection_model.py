from abc import ABC

import timm
import torch
from torch import nn

from .detr import DetrHead
from .neck import Neck, LastFeaturesOnlyNeck, ConcatFeaturesNeck, BiFpnNeck


class BackboneNeckHeadNetwork(nn.Module):
    def __init__(self, backbone: nn.Module, neck: Neck, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, images: torch.Tensor, *head_args) -> list[torch.Tensor]:
        features = self.backbone(images)
        features = self.neck(features)
        return self.head(features, *head_args)


class EfficientVitDetr(nn.Module):
    def __init__(self, number_of_classes: int):
        super().__init__()
        backbone = timm.models.fastvit_t8(pretrained=True, features_only=True)
        features = backbone.feature_info.channels()
        self.model = BackboneNeckHeadNetwork(
            timm.models.fastvit_t8(pretrained=True, features_only=True),
            # LastFeaturesOnlyNeck(backbone.feature_info),
            ConcatFeaturesNeck(backbone.feature_info),
            # BiFpnNeck(backbone.feature_info, 3, features[-1]),
            DetrHead(number_of_classes, features[-1], 2),
        )

    def forward(self, images: torch.Tensor, *head_args) -> torch.Tensor:
        return self.model(images, *head_args)
