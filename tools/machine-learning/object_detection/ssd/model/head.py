import torch
from torch import nn
import torch.nn.functional as F


class BboxRegressionHead(nn.Module):
    def __init__(self, feature_size: int):
        super().__init__()
        self.regressor = nn.Linear(feature_size, 4)

    def forward(self, features: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        bbox = self.regressor(features)

        # assume bbox is xyxy format, offset is xy
        bbox[:, :2] += offset
        bbox[:, 2:] += offset

        return bbox


class ClassPredictorHead(nn.Module):
    def __init__(self, feature_size: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feature_size, num_classes)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight.original0.fill_(1)
        # Enforce unit norm for last layer
        self.last_layer.weight.original0.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)
