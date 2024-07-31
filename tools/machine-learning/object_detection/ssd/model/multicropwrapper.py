import torch
from torch import nn
import torch.nn.functional as F

from ssd.utils import assert_ndim


class MulticropWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, xs: list[torch.Tensor] | torch.Tensor):
        if not isinstance(xs, list):
            assert_ndim(xs, 3)
            xs = [xs]
        representations = [self.backbone(x) for x in xs]
        projections = [self.head(x) for x in representations]
        return projections
