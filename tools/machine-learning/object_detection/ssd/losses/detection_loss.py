from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import nn
import torch

@dataclass
class LossValues:
    classification_loss: torch.Tensor
    box_regression_loss: torch.Tensor

    def __post_init__(self):
        if not self.classification_loss.isfinite().all():
            raise ValueError("Classification loss is not finite")
        if not self.box_regression_loss.isfinite().all():
            raise ValueError("Box regression loss is not finite")

    def combine(
        self, lambdas: torch.Tensor | list[float] | None = None
    ) -> torch.Tensor:
        if lambdas is None:
            lambdas = torch.ones(2, device=self.classification_loss.device)
        lambdas = torch.as_tensor(lambdas, device=self.classification_loss.device)

        losses = torch.stack([self.classification_loss, self.box_regression_loss])
        return torch.nansum(losses * lambdas)

class DetectionLoss(ABC, nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, predictions: list[torch.Tensor] | torch.Tensor, targets: list[dict], *args, **kwargs) -> LossValues:
        pass
