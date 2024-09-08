from dataclasses import dataclass

import torch
from ssd.utils import assert_ndim, assert_shape
from torch import nn


@dataclass
class MatchingResult[T]:
    assigned_boxes: T
    assigned_classes: T


class TaskAlignedDetections(nn.Module):
    def forward_single(
        self, query_points: torch.Tensor, boxes: torch.Tensor, classes: torch.Tensor
    ) -> MatchingResult[torch.Tensor]:
        assert_ndim(query_points, 2)  # M, 2
        assert_ndim(boxes, 2)  # N, 4
        assert_ndim(classes, 1)  # N

        sentinel_box = torch.zeros((1, 4), device=boxes.device)
        sentinel_class = torch.zeros(1, dtype=torch.long, device=classes.device)

        boxes = torch.cat([sentinel_box, boxes])
        classes = torch.cat([sentinel_class, classes])

        M, _ = query_points.shape
        N = classes.numel()
        min, max = boxes[:, :2], boxes[:, 2:]
        areas = (max - min).prod(dim=1, keepdim=True)

        # is_larget and is_smaller are N, M
        is_larger = (min.unsqueeze(1) <= query_points).all(dim=-1)
        is_smaller = (max.unsqueeze(1) >= query_points).all(dim=-1)
        points_in_box = is_larger & is_smaller
        assert_shape(points_in_box, (N, M))

        # If the point intersects the box, its area is filled, otherwise it is filled with inf
        area_selector = torch.where(
            points_in_box, areas, torch.zeros_like(areas).fill_(torch.inf)
        )

        # Select smallest area for each query point, if point is not in any box, argmin will be zero, which is sentinel box
        selected_boxes = torch.argmin(area_selector, dim=0)
        assert_shape(selected_boxes, (M,))

        return MatchingResult(
            boxes[selected_boxes],
            classes[selected_boxes],
        )

    @torch.no_grad()
    def forward(
        self,
        query_points: list[torch.Tensor] | torch.Tensor,
        boxes: list[torch.Tensor],
        classes: list[torch.Tensor],
    ) -> MatchingResult[torch.Tensor]:
        gt_boxes = []
        gt_classes = []

        if isinstance(query_points, torch.Tensor) and query_points.ndim == 2:
            # Query points are equal for all images
            query_points = [query_points for _ in boxes]

        for points_in_image, boxes_in_image, classes_in_image in zip(
            query_points, boxes, classes
        ):
            assert_ndim(points_in_image, 2)
            assert_ndim(boxes_in_image, 2)
            assert_ndim(classes_in_image, 1)

            result = self.forward_single(
                points_in_image, boxes_in_image, classes_in_image
            )

            gt_boxes.append(result.assigned_boxes)
            gt_classes.append(result.assigned_classes)

        return MatchingResult(
            torch.stack(gt_boxes),
            torch.stack(gt_classes),
        )


def compute_anchors(
    width: int, height: int, xrange: tuple[float, float] = (0, 1), yrange: tuple[float, float] = (0, 1)
) -> torch.Tensor:
    min_x, max_x = xrange
    min_y, max_y = yrange

    x_offset = (max_x - min_x) / (2 * (width - 1))
    y_offset = (max_y - min_y) / (2 * (height - 1))

    center_xs = torch.linspace(min_x + x_offset, max_x - x_offset, width)
    center_ys = torch.linspace(min_y + y_offset, max_y - y_offset, height)

    grid = torch.meshgrid(center_xs, center_ys, indexing="xy")
    return torch.stack(grid, dim=-1).flatten(0, 1)
