import plotly.express as px
import plotly.graph_objects as go
import torch
from ssd.task_aligner import TaskAlignedDetections
from ssd.utils import postprocess_output


def show_train_example(
    image: torch.Tensor,
    targets: dict[str, torch.Tensor],
    class_list: list[str],
    query_points: torch.Tensor,
    output: torch.Tensor | None = None,
) -> go.Figure:
    image = (image - image.min()) / (image.max() - image.min())
    image = image.permute(1, 2, 0).cpu().numpy()
    H, W, _ = image.shape

    boxes = targets["boxes"]
    classes = targets["classes"]

    fig = px.imshow(image)
    for box, cls in zip(boxes, classes):
        [x_min, y_min, x_max, y_max] = box.tolist()
        fig.add_shape(
            type="rect",
            x0=x_min * W,
            y0=y_min * H,
            x1=x_max * W,
            y1=y_max * H,
            line=dict(color="blue"),
            fillcolor=None,
        )
        fig.add_annotation(
            x=x_min * W,
            y=y_min * H,
            text=class_list[cls - 1],
            showarrow=False,
            yanchor="bottom",
            xanchor="left",
        )

    query_classes = (
        TaskAlignedDetections()
        .forward_single(query_points, boxes, classes)
        .assigned_classes
    )
    xs = query_points[:, 0] * W
    ys = query_points[:, 1] * H
    fig.add_scatter(
        x=xs,
        y=ys,
        mode="markers",
        marker=dict(size=10, color="red"),
    )

    if output is not None:
        predicted_boxes, predicted_classes = postprocess_output(output)

        for box, cls in zip(predicted_boxes, predicted_classes):
            [x_min, y_min, x_max, y_max] = box.tolist()
            fig.add_shape(
                type="rect",
                x0=x_min * W,
                y0=y_min * H,
                x1=x_max * W,
                y1=y_max * H,
                line=dict(color="green", dash="dot"),
                fillcolor=None,
            )
            fig.add_annotation(
                x=x_min * W,
                y=y_min * H,
                text=class_list[cls - 1],
                showarrow=False,
                yanchor="bottom",
                xanchor="left",
            )

    return fig
