from dataclasses import asdict, dataclass
from typing import Any, Literal

from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.scheduler.scheduler import Scheduler
from torch.optim.optimizer import Optimizer


@dataclass
class Config:
    def dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizerConfig(Config):
    learning_rate: float = 0.001
    weight_decay: float = 0.04
    momentum: float = 0.9
    optimizer: Literal["sgd", "adam", "adamw"] = "adamw"


def build_optimizer(model, optimizer_config: OptimizerConfig) -> Optimizer:
    return create_optimizer_v2(
        model,
        opt=optimizer_config.optimizer,
        lr=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
        momentum=optimizer_config.momentum,
    )


@dataclass
class SchedulerConfig(Config):
    schedule: Literal["step", "cosine"] = "cosine"
    number_of_epochs: int = 300
    min_learning_rate: float = 1e-6
    warmump_learning_rate: float = 1e-6
    warmup_epochs: int = 15


def build_scheduler(optimizer, scheduler_config: SchedulerConfig) -> Scheduler:
    scheduler = create_scheduler_v2(
        optimizer,
        num_epochs=scheduler_config.number_of_epochs,
        min_lr=scheduler_config.min_learning_rate,
        warmup_lr=scheduler_config.warmump_learning_rate,
        warmup_epochs=scheduler_config.warmup_epochs,
    )[0]
    if scheduler is None:
        raise ValueError("Scheduler is None")
    return scheduler


@dataclass
class DataLoaderConfig(Config):
    batch_size: int = 64
    num_workers: int = 4
