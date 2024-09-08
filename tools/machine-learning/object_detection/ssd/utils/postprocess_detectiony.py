import torch
from torchvision.ops import batched_nms


def postprocess_output(
    output: torch.Tensor, iou_threshold: float = 0.7
) -> tuple[torch.Tensor, torch.Tensor]:
    boxes_and_classes_size = output.size(-1)
    output = output.view(-1, boxes_and_classes_size)

    all_scores = output[..., 4:]

    # Select non-background predictions
    selector = all_scores.argmax(dim=-1) > 0

    scores = output[selector, 4:]
    scores, class_indices = scores.max(dim=-1)
    boxes = output[selector, :4]

    selected_boxes = batched_nms(
        boxes, scores, class_indices, iou_threshold=iou_threshold
    )

    return boxes[selected_boxes], class_indices[selected_boxes]
