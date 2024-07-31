import torch
from ssd.utils import assert_ndim, assert_shape
from torch import nn
from torchvision.ops import complete_box_iou


class OneToOneAligner(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, predictions: torch.Tensor, ground_truths: torch.Tensor):
        assert_ndim(predictions, 2)  # N,4
        assert_ndim(ground_truths, 2)  # M,4

        score_matrix = complete_box_iou(predictions, ground_truths)  # N,M
        target_truth = torch.argmax(score_matrix, dim=1)  # N

        return target_truth
