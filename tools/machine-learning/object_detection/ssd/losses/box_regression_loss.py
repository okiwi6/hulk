import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops.ciou_loss import complete_box_iou_loss


class BoxRegressionLoss(nn.Module):
    def __init__(self, image_size: int | tuple[int, int] = 224):
        super().__init__()
        if not isinstance(image_size, tuple):
            width, height = (
                image_size,
                image_size,
            )
        else:
            width, height = image_size
        self.scaler = torch.tensor([width, height, width, height], dtype=torch.float32)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Assumes xyxy format for predictions and targets
        self.scaler = self.scaler.to(predictions.device)
        return box_regression_loss(self.scaler * predictions, self.scaler * targets)


def box_regression_loss(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    assert predictions.ndim == 2
    assert targets.ndim == 2
    assert predictions.shape == targets.shape

    l1_loss = F.smooth_l1_loss(predictions, targets, reduction="mean")
    ciou_loss = complete_box_iou_loss(predictions, targets, reduction="mean")

    return l1_loss + ciou_loss
