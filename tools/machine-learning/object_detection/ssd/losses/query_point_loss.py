from dataclasses import dataclass

import torch
from ssd.utils import assert_ndim, assert_shape
from ssd.task_aligner import TaskAlignedDetections
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence

from .focal_loss import CategoricalFocalLoss
from .box_regression_loss import BoxRegressionLoss


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


class QueryPointLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.classification_criterion = CategoricalFocalLoss(label_smoothing=0.05)
        self.classification_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.box_regression_criterion = BoxRegressionLoss()
        self.matcher = TaskAlignedDetections()

    def forward(
        self,
        all_predictions: torch.Tensor,
        all_targets: list[dict],
        sample_points: torch.Tensor,
    ) -> LossValues:
        device = all_predictions.device
        assert all_predictions.isfinite().all()
        assert all_predictions.ndim == 3

        predicted_boxes = all_predictions[..., :4].flatten(0, 1)  # Batch * Queries, 4
        predicted_classes = all_predictions[..., 4:].flatten(
            0, 1
        )  # Batch * Queries, Num Classes + 1

        classification_losses = []
        box_losses = []

        gt_boxes = [target["boxes"] for target in all_targets]
        gt_classes = [target["classes"] for target in all_targets]

        matching_result = self.matcher.forward(sample_points, gt_boxes, gt_classes)
        assigned_boxes = matching_result.assigned_boxes.flatten(0, 1)
        assigned_classes = matching_result.assigned_classes.flatten(0, 1)

        classification_loss = self.classification_criterion(
            predicted_classes, assigned_classes
        )
        is_not_background = assigned_classes != 0

        # If all assigned classes are background, the box regression loss is 0
        if not is_not_background.any():
            return LossValues(classification_loss, torch.tensor(0.0, device=device))

        box_regression_loss = self.box_regression_criterion(
            predicted_boxes[is_not_background], assigned_boxes[is_not_background]
        )

        return LossValues(
            classification_loss, box_regression_loss / is_not_background.sum()
        )
