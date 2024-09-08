from dataclasses import dataclass

import torch
from ssd.task_aligner import TaskAlignedDetections
from ssd.utils import assert_ndim, assert_shape
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence

from .box_regression_loss import BoxRegressionLoss
from .focal_loss import CategoricalFocalLoss
from .detection_loss import LossValues, DetectionLoss

class QueryPointLoss(DetectionLoss):
    def __init__(self):
        super().__init__()
        self.classification_criterion = CategoricalFocalLoss(label_smoothing=0.05)
        # self.classification_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
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

        predicted_boxes = all_predictions[..., :4].flatten(0, -2)  # Batch * Queries, 4
        predicted_classes = all_predictions[..., 4:].flatten(
            0, -2
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
