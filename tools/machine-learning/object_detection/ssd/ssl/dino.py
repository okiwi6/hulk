from copy import deepcopy

import torch
from ssd.model import DINOHead, MulticropWrapper
from torch import nn


def representation(model: nn.Module, crops: list[torch.Tensor]) -> list[torch.Tensor]:
    return [model(crop) for crop in crops]


class DinoGym:
    def __init__(self, student_model: nn.Module, embedding_dimension: int):
        self.student_model = MulticropWrapper(
            student_model, DINOHead(embedding_dimension, 8192)
        )
        self.teacher_model = deepcopy(self.student_model)

    def training_step(
        self, local_crops: list[torch.Tensor], global_crops: list[torch.Tensor]
    ) -> torch.Tensor:
        local_student_representations = self.student_model(local_crops)
        global_student_representations = self.student_model(global_crops)

        with torch.no_grad():
            global_teacher_representations = self.teacher_model(global_crops)
