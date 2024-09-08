from collections.abc import Callable
from random import randint

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from rich.progress import MofNCompleteColumn, Progress
from ssd.dataset import COCODataset, DetectionDataset, VocDataset
from ssd.losses import QueryPointLoss
from ssd.model import EfficientVitDetr
from ssd.model.detection_model import EfficientVitSsd
from ssd.task_aligner import TaskAlignedDetections
from ssd.task_aligner.task_aligner import compute_anchors
from ssd.utils import (DataLoaderConfig, OptimizerConfig, SchedulerConfig,
                       assert_ndim, bar, build_optimizer, build_scheduler,
                       default_progress)
from ssd.utils.accelerate_setup import build_dataloaders, datasets
from ssd.visualization import show_train_example
from timm.scheduler.scheduler import Scheduler
from torch import nn
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, random_split
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import MulticlassF1Score

wandb.require("core")

def update_f1(
    outputs: torch.Tensor,
    all_targets: list[dict[str, torch.Tensor]],
    sample_points: torch.Tensor,
    f1_score: MulticlassF1Score,
):
    device = outputs.device
    matcher = TaskAlignedDetections()

    predicted_classes = outputs[..., 4:].flatten(0, -2)
    gt_boxes = [target["boxes"] for target in all_targets]
    gt_classes = [target["classes"] for target in all_targets]

    result = matcher.forward(sample_points, gt_boxes, gt_classes)
    selected_gt_classes = result.assigned_classes.flatten(0, 1)

    f1_score.update(predicted_classes, selected_gt_classes)


def main():
    dataloader_config = DataLoaderConfig()
    optimizer_config = OptimizerConfig()
    scheduler_config = SchedulerConfig()

    run = wandb.init(
        project="voc-query-detection",
        config={
            "dataloader": dataloader_config.dict(),
            "optimizer": optimizer_config.dict(),
            "scheduler": scheduler_config.dict(),
        },
        mode = "disabled",
    )

    accelerator = Accelerator()
    device = accelerator.device
    print("Running on", device)

    train_set, val_set = datasets("voc")
    train_loader, val_loader = build_dataloaders(dataloader_config, train_set, val_set)

    # Add one background class (index: 0)
    model = EfficientVitSsd(train_set.n_classes() + 1)
    model.summary()

    criterion = QueryPointLoss()
    optimizer = build_optimizer(model, optimizer_config)
    scheduler = build_scheduler(optimizer, scheduler_config)

    (
        model,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
        criterion,
    ) = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler, criterion
    )

    global_train_step = 0
    with default_progress() as progress:
        for epoch in bar(
            range(scheduler_config.number_of_epochs), progress, "[cyan]Epochs"
        ):
            train_score = MulticlassF1Score(
                train_set.n_classes() + 1, average="none"
            ).to(device)
            for images, targets in bar(train_loader, progress, "[red]Fit"):
                optimizer.zero_grad()
                outputs = model(images)[0]
                _, h, w, _ = outputs.shape
                anchors = compute_anchors(w, h).to(device)
                losses = criterion(outputs, targets, anchors)
                update_f1(outputs, targets, anchors, train_score)

                combined_loss = losses.combine([1.0, 1.0])
                assert combined_loss.isfinite(), "Loss is not finite"
                wandb.log(
                    {
                        "train/loss": combined_loss.item(),
                        "train/classification_loss": losses.classification_loss.item(),
                        "train/box_regression_loss": losses.box_regression_loss.item(),
                    },
                    step=global_train_step,
                )
                accelerator.backward(combined_loss)
                accelerator.clip_grad_norm_(model.parameters(), 20.0)
                optimizer.step()
                scheduler.step(epoch)
                global_train_step += 1

            classes = ["background"] + train_set.classes()
            class_scores = {
                name: score for name, score in zip(classes, train_score.compute())
            }
            wandb.log(
                {f"train/f1/{name}": score for name, score in class_scores.items()}
                | {
                    "epoch": epoch,
                    "learning_rate": scheduler._get_lr(epoch)[0],
                    "train/f1": train_score.compute().mean(),
                },
                step=global_train_step,
            )

            val_score = MulticlassF1Score(train_set.n_classes() + 1, average="none").to(
                device
            )
            mean_classification_loss = MeanMetric().to(device)
            mean_regression_loss = MeanMetric().to(device)

            with torch.inference_mode():
                for images, targets in bar(val_loader, progress, "[green]Val"):
                    outputs = model(images)[0]
                    _, h, w, _ = outputs.shape
                    anchors = compute_anchors(w, h).to(device)
                    losses = criterion(outputs, targets, anchors)
                    update_f1(outputs, targets, anchors, val_score)

                    mean_classification_loss.update(losses.classification_loss)
                    mean_regression_loss.update(losses.box_regression_loss)

                # Log one random image
                index = randint(0, len(val_set))
                image, target = val_set[index]
                image = image.unsqueeze(0).to(device)
                output = model(image)[0]
                _, h, w, _ = output.shape
                anchors = compute_anchors(w, h)
            fig = show_train_example(
                image[0].cpu(),
                target,
                train_set.classes(),
                anchors,
                output.cpu(),
            )
            mean_classification_loss = mean_classification_loss.compute()
            mean_regression_loss = mean_regression_loss.compute()
            wandb.log(
                {
                    "val/f1": val_score.compute().mean(),
                    "val/example": fig,
                    "val/mean_classification_loss": mean_classification_loss,
                    "val/mean_regression_loss": mean_regression_loss,
                    "val/mean_combined_loss": mean_classification_loss
                    + mean_regression_loss,
                },
                step=global_train_step,
            )

            classes = ["background"] + train_set.classes()
            class_scores = {
                name: score for name, score in zip(classes, val_score.compute())
            }
            wandb.log(
                {f"val/f1/{name}": score for name, score in class_scores.items()},
                step=global_train_step,
            )
        model.export_onnx("ssd_embedding.onnx", "ssd_detection.onnx")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
