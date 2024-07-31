from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torchvision.datasets import VOCDetection
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

from .dataset import DetectionDataset

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def bounding_box_to_tensor(box: dict, width: int, height: int) -> torch.Tensor:
    bbox = box["bndbox"]

    xmin = int(bbox["xmin"])
    ymin = int(bbox["ymin"])
    xmax = int(bbox["xmax"])
    ymax = int(bbox["ymax"])

    return torch.tensor(
        [xmin / width, ymin / height, xmax / width, ymax / height], dtype=torch.float32
    )


class VocDataset(DetectionDataset):
    def __init__(self, variant: Literal["train", "val"]):
        path = Path("datasets/VOCdevkit")
        transform = v2.Compose(
            [
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        if path.exists():
            self.dataset = VOCDetection(
                "datasets",
                download=False,
                year="2012",
                transform=transform,
                image_set=variant,
            )
        else:
            self.dataset = VOCDetection(
                "datasets", download=True, transform=transform, image_set=variant
            )

    def n_classes(self) -> int:
        return len(VOC_CLASSES)

    def classes(self) -> list[str]:
        return VOC_CLASSES

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image, labels = self.dataset[index]
        _, height, width = image.shape

        annotations = labels["annotation"]["object"]
        boxes = [bounding_box_to_tensor(box, width, height) for box in annotations]
        classes = [VOC_CLASSES.index(box["name"]) + 1 for box in annotations]

        targets = {
            "boxes": torch.stack(boxes),
            "classes": torch.tensor(classes, dtype=torch.long),
        }

        return F.resize(image, [224, 224]), targets

    def collate_fn(self, batch: list[tuple]) -> tuple:
        images = torch.stack([image for image, _ in batch])
        targets = [target for _, target in batch]

        return images, targets
