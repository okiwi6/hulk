import torch
import torch.nn.functional as F
from ssd.utils import assert_ndim
from torch import nn


class CategoricalFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return categorical_focal_loss(
            predictions, targets, self.alpha, self.gamma, self.reduction
        )


def categorical_focal_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    assert_ndim(predictions, 2)
    assert_ndim(targets, 1)

    logpt = F.log_softmax(predictions, dim=-1)

    logpt = alpha * (1 - logpt.exp()) ** gamma * logpt
    return F.nll_loss(logpt, targets, reduction=reduction)
