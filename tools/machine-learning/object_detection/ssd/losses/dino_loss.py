import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from ssd.utils import assert_ndim


@dataclass
class DINOLossConfiguration:
    number_of_crops: int = 7
    student_temperature: float = 0.1
    teacher_temperature: float = 0.04
    center_momentum: float = 0.9


class DINOLoss(nn.Module):
    def __init__(self, config: DINOLossConfiguration, out_dim: int):
        self.config = config
        self.is_initialized = False
        self.register_buffer("center_mean", torch.zeros(1, out_dim))

    def forward(
        self, student_output: torch.Tensor, teacher_output: torch.Tensor
    ) -> torch.Tensor:
        assert_ndim(student_output, 3)
        assert_ndim(teacher_output, 3)

        student_output = student_output / self.config.student_temperature

        teacher_output = teacher_output.detach()
        teacher_representations = self._prepare_teacher_output(teacher_output)

        left, right = torch.combinations(
            torch.arange(self.config.number_of_crops)
        ).unbind(-1)
        return F.cross_entropy(student_output[left], teacher_representations[right])

    @torch.no_grad()
    def _center_update(self, teacher_output: torch.Tensor):
        new_center = teacher_output.mean(dim=0, keepdim=True)
        alpha = self.config.center_momentum

        if self.is_initialized:
            # EMA center update
            self.center = self.center * alpha + new_center * (1.0 - alpha)
        else:
            self.center = new_center
            self.is_initialized = True

    @torch.no_grad()
    def _prepare_teacher_output(self, teacher_output: torch.Tensor) -> torch.Tensor:
        if not self.is_initialized:
            center = teacher_output.mean(dim=0, keepdim=True)
        else:
            center = self.center

        return F.softmax(
            (teacher_output - center) / self.config.teacher_temperature, dim=-1
        )
