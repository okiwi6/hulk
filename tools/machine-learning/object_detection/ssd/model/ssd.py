import torch
import torch.nn.functional as F
from ssd.task_aligner import compute_anchors
from torch import nn

from .head import BboxRegressionHead, ClassPredictorHead


class SSDHead(nn.Module):
    def __init__(self, num_classes: int, feature_size: int):
        super().__init__()
        self.regressor = BboxRegressionHead(feature_size)
        self.classifier = ClassPredictorHead(feature_size, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        n, c, h, w = x.shape
        x = x.view(n, c, h * w)
        x = x.permute(0, 2, 1).contiguous()  # n, h * w, c

        offsets = compute_anchors(w, h).to(x.device)
        bbox = self.regressor(x, offsets)
        classes = self.classifier(x)

        return torch.cat([bbox, classes], dim=-1).view(n, h, w, 4 + self.num_classes)
