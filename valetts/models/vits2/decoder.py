"""
Flow-based Decoder para VITS2 - Implementação completa.

Este módulo implementa o decoder baseado em flows do VITS2, incluindo:
- Normalizing Flows com coupling layers
- WaveNet residual blocks
- Upsampling layers para geração de waveform
- Invertible transformations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

from ..base.layers import Conv1d, ResidualBlock
from .config import VITS2Config


class Generator(nn.Module):
    """
    Generator/Decoder do VITS2 baseado em Normalizing Flows.

    Converte representações latentes em waveforms através de
    transformações invertíveis e upsampling.
    """

    def __init__(self, config: VITS2Config):
        super().__init__()
        self.config = config

        # Upsampling factors
        self.upsample_rates = config.upsample_rates
        self.upsample_kernel_sizes = config.upsample_kernel_sizes

        # Normalizing flows
        self.flows = NormalizingFlows(
            channels=config.latent_dim,
            hidden_channels=config.flow_hidden_channels,
            kernel_size=config.flow_kernel_size,
            dilation_rate=config.flow_dilation_rate,
            n_blocks=config.n_flow_blocks,
            n_layers=config.n_flow_layers
        )

        # Convolution pré-upsampling
        self.pre_conv = Conv1d(
            in_channels=config.latent_dim,
            out_channels=config.decoder_hidden_dim,
            kernel_size=7,
            padding=3,
            activation='leaky_relu'
        )

        # Upsampling layers
        self.upsample_layers = nn.ModuleList()

        # Residual blocks entre upsampling
        self.residual_blocks = nn.ModuleList()

        current_channels = config.decoder_hidden_dim

        for i, (upsample_rate, kernel_size) in enumerate(
            zip(self.upsample_rates, self.upsample_kernel_sizes)
        ):
            # Upsampling layer
            self.upsample_layers.append(
                nn.ConvTranspose1d(
                    current_channels,
                    current_channels // 2,
                    kernel_size,
                    upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2
                )
            )

            current_channels = current_channels // 2

            # Residual blocks after upsampling
            residual_block = nn.ModuleList()
            for j in range(config.resblock_layers):
                residual_block.append(
                    WaveNetResidualBlock(
                        channels=current_channels,
                        kernel_size=config.resblock_kernel_sizes[j % len(config.resblock_kernel_sizes)],
                        dilation=config.resblock_dilations[j % len(config.resblock_dilations)]
                    )
                )
            self.residual_blocks.append(residual_block)

        # Convolução final para waveform
        self.post_conv = Conv1d(
            in_channels=current_channels,
            out_channels=1,
            kernel_size=7,
            padding=3,
            activation='tanh'
        )

    def forward(
        self,
        z: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False
    ) -> torch.Tensor:
        """
        Forward pass do generator.

        Args:
            z: Representações latentes [batch_size, latent_dim, seq_len]
            g: Global conditioning (speaker embedding, etc.)
            reverse: Se True, aplica flows de forma reversa

        Returns:
            Waveform gerado [batch_size, 1, audio_len]
        """
        # Aplicar normalizing flows
        if reverse:
            z = self.flows(z, reverse=True)
        else:
            z = self.flows(z, reverse=False)

        # Convolução pré-upsampling
        x = self.pre_conv(z)

        # Adicionar conditioning global se fornecido
        if g is not None:
            x = x + g.unsqueeze(-1)

        # Upsampling e residual blocks
        for upsample_layer, residual_blocks in zip(
            self.upsample_layers, self.residual_blocks
        ):
            # Upsampling
            x = F.leaky_relu(upsample_layer(x), 0.1)

            # Residual blocks
            for residual_block in residual_blocks:
                x = residual_block(x)

        # Convolução final
        waveform = self.post_conv(x)

        return waveform

    def inference(self, z: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Inferência otimizada."""
        return self.forward(z, g, reverse=True)


class NormalizingFlows(nn.Module):
    """
    Normalizing Flows implementation para VITS2.

    Série de transformações invertíveis que mapeiam entre
    distribuições latentes e features acústicas.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_blocks: int,
        n_layers: int
    ):
        super().__init__()

        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_blocks = n_blocks
        self.n_layers = n_layers

        # Flow blocks
        self.flows = nn.ModuleList()

        for i in range(n_blocks):
            self.flows.append(
                CouplingLayer(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    n_layers=n_layers,
                    dilation_rate=dilation_rate,
                    flip=i % 2 == 0  # Alternar entre transformações
                )
            )

        # Learnable prior
        self.log_prob_prior = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        reverse: bool = False
    ) -> torch.Tensor:
        """
        Aplica flows de forma direta ou reversa.

        Args:
            x: Input tensor [batch_size, channels, seq_len]
            x_mask: Máscara opcional
            reverse: Se True, aplica transformação reversa

        Returns:
            Tensor transformado
        """
        if not reverse:
            # Forward: x -> z
            for flow in self.flows:
                x = flow(x, x_mask, reverse=False)
        else:
            # Reverse: z -> x
            for flow in reversed(self.flows):
                x = flow(x, x_mask, reverse=True)

        return x


class CouplingLayer(nn.Module):
    """
    Coupling layer para Normalizing Flows.

    Implementa transformação invertível baseada em
    affine coupling functions.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        n_layers: int,
        dilation_rate: int = 1,
        flip: bool = False
    ):
        super().__init__()

        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.flip = flip

        # Split channels in half
        self.half_channels = channels // 2

        # Transformation network
        self.transform_net = WaveNet(
            in_channels=self.half_channels,
            hidden_channels=hidden_channels,
            out_channels=self.half_channels * 2,  # Para scale e translation
            kernel_size=kernel_size,
            n_layers=n_layers,
            dilation_rate=dilation_rate
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        reverse: bool = False
    ) -> torch.Tensor:
        """
        Aplica transformação coupling.

        Args:
            x: Input [batch_size, channels, seq_len]
            x_mask: Máscara opcional
            reverse: Direção da transformação

        Returns:
            Tensor transformado
        """
        # Split input em duas partes
        if self.flip:
            x0, x1 = x[:, self.half_channels:], x[:, :self.half_channels]
        else:
            x0, x1 = x[:, :self.half_channels], x[:, self.half_channels:]

        # Predizer scale e translation
        h = self.transform_net(x0, x_mask)
        shift = h[:, :self.half_channels]
        scale = h[:, self.half_channels:]
        scale = torch.sigmoid(scale + 2.0)  # Garantir estabilidade

        if not reverse:
            # Forward: x1 = x1 * scale + shift
            x1 = x1 * scale + shift
        else:
            # Reverse: x1 = (x1 - shift) / scale
            x1 = (x1 - shift) / scale

        # Concatenar de volta
        if self.flip:
            x = torch.cat([x1, x0], dim=1)
        else:
            x = torch.cat([x0, x1], dim=1)

        return x


class WaveNet(nn.Module):
    """
    WaveNet-style network para coupling layers.

    Usa dilated convolutions para capturar dependências
    de longo alcance.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        n_layers: int,
        dilation_rate: int = 2
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers

        # Input projection
        self.start = Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1
        )

        # Dilated convolution layers
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            dilation = dilation_rate ** i
            self.layers.append(
                WaveNetLayer(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation
                )
            )

        # Output projection
        self.end = Conv1d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass do WaveNet."""
        x = self.start(x)

        # Dilated convolution layers
        for layer in self.layers:
            x = layer(x, x_mask)

        x = self.end(x)

        if x_mask is not None:
            x = x * x_mask

        return x


class WaveNetLayer(nn.Module):
    """Camada individual do WaveNet com gated activation."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()

        self.dilation = dilation
        self.conv = nn.Conv1d(
            channels,
            channels * 2,
            kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation
        )

        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward com gated activation."""
        residual = x

        x = self.conv(x)

        # Gated activation
        x_tanh, x_sigmoid = x.chunk(2, dim=1)
        x = torch.tanh(x_tanh) * torch.sigmoid(x_sigmoid)

        x = self.proj(x)

        # Residual connection
        x = x + residual

        if x_mask is not None:
            x = x * x_mask

        return x


class WaveNetResidualBlock(nn.Module):
    """
    Bloco residual WaveNet para o decoder.

    Usa dilated convolutions para modelar dependências
    temporais de longo alcance.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()

        self.conv1 = Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            activation='leaky_relu'
        )

        self.conv2 = Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            activation='leaky_relu'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass com residual connection."""
        residual = x

        x = self.conv1(x)
        x = self.conv2(x)

        return x + residual
