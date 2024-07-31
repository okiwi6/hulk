import torch
from ssd.utils import assert_ndim
from torch import nn

from .head import BboxRegressionHead, ClassPredictorHead


class PositionalEncoding(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size
        self.sin_path = nn.Linear(2, output_size // 2)
        self.cos_path = nn.Linear(2, output_size // 2)

    def forward(self, points: torch.Tensor):
        shape = points.shape
        positional_encoding = torch.cat(
            [self.sin_path(points).sin(), self.cos_path(points).cos()], dim=-1
        )

        return positional_encoding.view(*shape[:-1], self.output_size)


class DetrHead(nn.Module):
    def __init__(self, num_classes: int, feature_size: int, nheads: int):
        super().__init__()
        self.num_classes = num_classes

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(feature_size, nheads, batch_first=True),
            1,
            nn.LayerNorm(feature_size),
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(feature_size, nheads, batch_first=True),
            1,
            nn.LayerNorm(feature_size),
        )
        self.positional_encoding = PositionalEncoding(feature_size)

        self.classifier = ClassPredictorHead(feature_size, num_classes)
        self.bbox_regressor = BboxRegressionHead(feature_size)

    def forward(
        self, features: torch.Tensor, sample_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: torch.Tensor of shape (N, C, H, W)
            queries: torch.Tensor of shape (N, L, 2)

        Returns:
            torch.Tensor of shape (N, L, 4 + C)
        """
        assert_ndim(features, 4)
        assert_ndim(sample_points, 3)

        N, C, H, W = features.shape
        tokens = features.view(N, C, H * W)

        # Make channels last
        tokens = tokens.permute(0, 2, 1)

        device = tokens.device
        x_offset = 0.5 / (W - 1)
        y_offset = 0.5 / (H - 1)
        xs, ys = torch.meshgrid(
            torch.linspace(x_offset, 1.0 - x_offset, W, device=device),
            torch.linspace(y_offset, 1.0 - y_offset, H, device=device),
            indexing="ij",
        )
        token_positions = torch.stack((xs, ys), dim=-1).view(-1, 2)
        token_positional_encodings = self.positional_encoding(token_positions)

        encodings = self.encoder(tokens + token_positional_encodings)

        queries = self.positional_encoding(sample_points)
        decodings = self.decoder(queries, encodings)

        return torch.stack(
            [
                self._regress_and_predict(decoding, offset)
                for decoding, offset in zip(decodings, sample_points)
            ]
        )

    def _regress_and_predict(
        self, decoding: torch.Tensor, sample_points: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            [
                self.bbox_regressor(decoding, sample_points),
                self.classifier(decoding),
            ],
            dim=-1,
        )
