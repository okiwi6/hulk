from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

from .dataset import DetectionDataset


def bounding_box_to_tensor(box: dict, width: int, height: int) -> torch.Tensor:
    [xmin, ymin, w, h] = box["bbox"]

    return torch.tensor(
        [xmin / width, ymin / height, (xmin + w) / width, (ymin + h) / height],
        dtype=torch.float32,
    )


class COCODataset(DetectionDataset):
    def __init__(
        self, variant: Literal["train", "val"], year: Literal["2017"] = "2017"
    ):
        path = Path("datasets/COCO")
        transform = v2.Compose(
            [
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        image_directory = path / f"{variant}{year}"
        annotation_file = path / f"annotations/instances_{variant}{year}.json"

        self.dataset = CocoDetection(image_directory, str(annotation_file), transform)

        self.id2class_index = self.dataset.coco.getCatIds()
        self.class_list = [
            x["name"] for x in self.dataset.coco.loadCats(self.id2class_index)
        ]

    def n_classes(self) -> int:
        return len(self.class_list)

    def classes(self) -> list[str]:
        return self.class_list

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image, labels = self.dataset[index]
        _, height, width = image.shape

        boxes = [bounding_box_to_tensor(box, width, height) for box in labels]
        if len(boxes) == 0:
            boxes = torch.zeros(0, 4, dtype=torch.float)
        else:
            boxes = torch.stack(boxes)
        classes = torch.tensor(
            [self.id2class_index.index(box["category_id"]) + 1 for box in labels],
            dtype=torch.long,
        )

        targets = {
            "boxes": boxes,
            "classes": classes,
        }

        return F.resize(image, [224, 224]), targets

    def collate_fn(self, batch: list[tuple]) -> tuple:
        images = torch.stack([image for image, _ in batch])
        targets = [target for _, target in batch]

        return images, targets
