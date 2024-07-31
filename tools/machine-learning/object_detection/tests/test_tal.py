import torch
from ssd.task_aligner.task_aligner import TaskAlignedDetections, compute_anchors
from ssd.utils import assert_ndim, assert_shape
from torch.testing import assert_close
from torchvision.tv_tensors import BoundingBoxes


def test_compute_anchors():
    anchors = compute_anchors((-2, -1), 21, (1, 2), 11)
    assert_shape(anchors, (21 * 11, 2))

    xs = anchors[:, 0]
    ys = anchors[:, 1]

    assert torch.all(xs > -2)
    assert torch.all(xs < -1)
    assert torch.all(ys > 1)
    assert torch.all(ys < 2)


def test_tal():
    prediction = torch.zeros(1, 10, 10, 4)
    boxes = [BoundingBoxes(torch.zeros(1, 4), format="xywh", canvas_size=(10, 10))]

    tal = TaskAlignedDetections()
    tal(prediction, boxes)
