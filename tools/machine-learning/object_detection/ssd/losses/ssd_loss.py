from typing_extensions import override
from urllib.parse import _QueryType
import torch

from ssd.losses.query_point_loss import QueryPointLoss
from ssd.task_aligner import TaskAlignedDetections, compute_anchors

from .detection_loss import DetectionLoss, LossValues
from .focal_loss import CategoricalFocalLoss
from .box_regression_loss import BoxRegressionLoss


class SsdLoss(QueryPointLoss):
    def __init__(self):
        self.classification_criterion = CategoricalFocalLoss(label_smoothing=0.05)
        # self.classification_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.box_regression_criterion = BoxRegressionLoss()
        self.matcher = TaskAlignedDetections()

    @override
    def forward(self, all_predictions: torch.Tensor, all_targets: list[dict], anchor_points: torch.Tensor) -> LossValues:
        return super().forward(all_predictions, all_targets, anchor_points)
