import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, number_of_heads: int):
        super().__init__()
        assert (
            d_model % number_of_heads == 0
        ), "Feature dimension is not divisible by number of heads"

        self.d_model = d_model
        self.number_of_heads = number_of_heads
        self.d_k = d_model // number_of_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, d_model = x.shape
        return x.view(
            batch_size, sequence_length, self.number_of_heads, self.d_k
        ).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _number_of_heads, sequence_length, d_k = x.shape
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.d_model)
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        Q = self.split_heads(self.W_q(q))
        K = self.split_heads(self.W_k(k))
        V = self.split_heads(self.W_v(v))

        attention_output = F.scaled_dot_product_attention(Q, K, V)
        return self.W_o(self.combine_heads(attention_output))


class FeedForwardLayer(nn.Sequential):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__(
            nn.Linear(d_model, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_model)
        )


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, number_of_heads: int, d_hidden: int, dropout: float = 0.2
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, number_of_heads)
        self.feed_forward = FeedForwardLayer(d_model, d_hidden)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_output = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attention_output))
        feed_forward_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(feed_forward_output))


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, number_of_heads: int, d_hidden: int, dropout: float = 0.2
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, number_of_heads)
        self.cross_attention = MultiHeadAttention(d_model, number_of_heads)
        self.feed_forward = FeedForwardLayer(d_model, d_hidden)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor, encoded_input: torch.Tensor) -> torch.Tensor:
        attention_output = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attention_output))
        cross_attention_output = self.cross_attention(x, encoded_input, encoded_input)
        x = self.norm2(x + self.dropout(cross_attention_output))
        feed_forward_output = self.feed_forward(x)
        return self.norm3(x + self.dropout(feed_forward_output))
