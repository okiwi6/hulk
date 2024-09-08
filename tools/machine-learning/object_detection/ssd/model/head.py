import torch
import torch.nn.functional as F
from torch import nn

from ssd.utils.assert_shape import assert_ndim


class BboxRegressionHead(nn.Module):
    def __init__(self, feature_size: int):
        super().__init__()
        self.regressor = nn.Linear(feature_size, 4)

    def forward(self, features: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        assert_ndim(features, 3)
        n, hw, d = features.shape
        bbox = self.regressor(features.view(-1, d)).view(n, hw, 4)

        # assume bbox is xyxy format, offset is xy
        bbox[..., :2] += offset
        bbox[..., 2:] += offset

        return bbox


class ClassPredictorHead(nn.Module):
    def __init__(self, feature_size: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feature_size, num_classes)
        )
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        assert_ndim(features, 3)
        n, hw, d = features.shape
        return self.classifier(features.view(-1, d)).view(n, hw, self.num_classes)


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
