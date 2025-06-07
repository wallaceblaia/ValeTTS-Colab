"""Base layers and building blocks for ValeTTS models."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Module):
    """1D Convolution with optional normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm: Optional[str] = None,
        activation: Optional[str] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Auto-calculate padding for 'same' padding
        if padding is None:
            padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # Normalization
        self.norm = None
        if norm == "batch":
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == "layer":
            self.norm = nn.LayerNorm(out_channels)
        elif norm == "group":
            self.norm = nn.GroupNorm(1, out_channels)

        # Activation
        self.activation = None
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish" or activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)

        if self.norm is not None:
            if isinstance(self.norm, nn.LayerNorm):
                x = x.transpose(1, 2)
                x = self.norm(x)
                x = x.transpose(1, 2)
            else:
                x = self.norm(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class ConvTranspose1d(nn.Module):
    """1D Transposed Convolution with optional normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        norm: Optional[str] = None,
        activation: Optional[str] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

        # Normalization
        self.norm = None
        if norm == "batch":
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == "layer":
            self.norm = nn.LayerNorm(out_channels)

        # Activation
        self.activation = None
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)

        if self.norm is not None:
            if isinstance(self.norm, nn.LayerNorm):
                x = x.transpose(1, 2)
                x = self.norm(x)
                x = x.transpose(1, 2)
            else:
                x = self.norm(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()

        self.conv1 = Conv1d(
            channels, channels, kernel_size,
            dilation=dilation, activation=activation, dropout=dropout
        )
        self.conv2 = Conv1d(
            channels, channels, kernel_size,
            dilation=dilation, dropout=dropout
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x + residual


class DilatedConvBlock(nn.Module):
    """Block of dilated convolutions with increasing dilation rates."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        num_layers: int = 4,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.layers.append(
                ResidualBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation=activation,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dilated layers."""
        for layer in self.layers:
            x = layer(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]

        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len = query.size(0), query.size(1)

        # Linear projections
        q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, v)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # Final linear projection
        output = self.w_o(context)

        return output, attention_weights


class FFN(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class LayerNorm(nn.Module):
    """Layer normalization with optional centering and scaling."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.norm(x)


class PositionalEncoding(nn.Module):
    """Learnable positional encoding."""

    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        self.embedding = nn.Embedding(max_length, d_model)
        self.max_length = max_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)
        if seq_len > self.max_length:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_length}")

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        pos_encoding = self.embedding(positions)

        return x + pos_encoding


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (fixed, not learnable)."""

    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            x + positional encoding
        """
        return x + self.pe[:x.size(1), :].transpose(0, 1)
