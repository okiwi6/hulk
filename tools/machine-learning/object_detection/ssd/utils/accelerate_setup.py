from typing import Literal

from accelerate import Accelerator
from ssd.dataset import COCODataset, DetectionDataset, VocDataset
from ssd.utils.config import (DataLoaderConfig, OptimizerConfig,
                              SchedulerConfig, build_optimizer,
                              build_scheduler)
from timm.scheduler.scheduler import Scheduler
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


def build_dataloaders(config: DataLoaderConfig, *datasets) -> tuple[DataLoader, ...]:
    return tuple(DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        shuffle=True,
    ) for dataset in datasets)

def datasets(
    dataset: Literal["voc", "coco"]
) -> tuple[DetectionDataset, DetectionDataset]:
    match dataset:
        case "voc":
            train_set = VocDataset("train")
            val_set = VocDataset("val")
        case "coco":
            train_set = COCODataset("train")
            val_set = COCODataset("val")

    return train_set, val_set
