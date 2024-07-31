import torch
import torch.nn.functional as F
from torch import nn


class MnistClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.model(x)
