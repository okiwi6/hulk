from abc import ABC
from pathlib import Path

from humanize import metric
import timm
import torch
from torch import nn

from .ssd import SSDHead
from .detr import DetrHead
from .neck import BiFpnNeck, ConcatFeaturesNeck, LastFeaturesOnlyNeck, Neck

class BackboneNeckHeadNetwork(nn.Module):
    def __init__(self, backbone: nn.Module, neck: Neck, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, images: torch.Tensor, *head_args) -> list[torch.Tensor]:
        features = self.backbone(images)
        features = self.neck(features)
        return [self.head(feature, *head_args) for feature in features]

    def _count_parameters(self, module: nn.Module) -> tuple[int, int]:
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

        return total, trainable

    def summary(self):
        total, trainable = self._count_parameters(self.backbone)
        print(f"{'Backbone':<10} {metric(total):<6} ({metric(trainable)})")
        total, trainable = self._count_parameters(self.neck)
        print(f"{'Neck':<10} {metric(total):<6} ({metric(trainable)})")
        total, trainable = self._count_parameters(self.head)
        print(f"{'Head':<10} {metric(total):<6} ({metric(trainable)})")
        total, trainable = self._count_parameters(self)
        print(f"{'Total':<10} {metric(total):<6} ({metric(trainable)})")

    @torch.no_grad()
    def export_onnx(
        self,
        embedding_model_path: str | Path,
        detection_model_path: str | Path,
        image_size: tuple[int, int] = (224, 224),
    ):
        example_input = torch.randn(1, 3, *image_size)
        example_points = torch.randn(1, 64, 2)

        embedding_model = nn.Sequential(self.backbone, self.neck).cpu()
        detection_model = self.head.cpu()

        example_embedding = embedding_model(example_input)
        example_detections = detection_model(example_embedding, example_points)

        torch.onnx.export(
            embedding_model,
            example_input,
            str(embedding_model_path),
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )
        torch.onnx.export(
            detection_model,
            (example_embedding, example_points),
            str(detection_model_path),
            input_names=["embedding", "queries"],
            output_names=["output"],
            dynamic_axes={"queries": {1: "n_queries"}, "output": {1: "n_queries"}},
            do_constant_folding=True,
        )


class EfficientVitDetr(BackboneNeckHeadNetwork):
    def __init__(self, number_of_classes: int):
        backbone = timm.models.fastvit_t8(pretrained=True, features_only=True)
        features = backbone.feature_info.channels()
        super().__init__(
            backbone,
            LastFeaturesOnlyNeck(backbone.feature_info),
            DetrHead(number_of_classes, features[-1], 2),
        )

class EfficientVitSsd(BackboneNeckHeadNetwork):
    def __init__(self, number_of_classes: int):
        backbone = timm.models.fastvit_t8(pretrained=True, features_only=True)
        features = backbone.feature_info.channels()
        super().__init__(
            backbone,
            LastFeaturesOnlyNeck(backbone.feature_info),
            SSDHead(number_of_classes, features[-1]),
        )
