"""
Discriminadores para VITS2 - Implementação completa.

Este módulo implementa os discriminadores do VITS2:
- Multi-Scale Discriminator (MSD) - Analisa audio em múltiplas escalas
- Multi-Period Discriminator (MPD) - Analisa patterns periódicos
- Loss functions adversariais
- Feature matching loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from ..base.layers import Conv1d
from .config import VITS2Config


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator (MSD) do VITS2.

    Analisa o áudio em múltiplas escalas temporais através
    de sub-discriminadores operando em diferentes resoluções.
    """

    def __init__(self, config: VITS2Config):
        super().__init__()
        self.config = config

        # Sub-discriminadores operando em diferentes escalas
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(config, scale=1),      # Original resolution
            ScaleDiscriminator(config, scale=2),      # 2x downsampled
            ScaleDiscriminator(config, scale=4),      # 4x downsampled
        ])

        # Pooling layers para downsampling
        self.pools = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
        ])

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Forward pass do MSD.

        Args:
            x: Waveform [batch_size, 1, seq_len]

        Returns:
            Tuple de:
                - outputs: Lista de outputs de cada sub-discriminador
                - feature_maps: Lista de feature maps de cada sub-discriminador
        """
        outputs = []
        feature_maps = []

        for pool, discriminator in zip(self.pools, self.discriminators):
            # Aplicar pooling para diferentes escalas
            x_scaled = pool(x)

            # Forward pass no sub-discriminador
            out, features = discriminator(x_scaled)

            outputs.append(out)
            feature_maps.append(features)

        return outputs, feature_maps


class ScaleDiscriminator(nn.Module):
    """
    Sub-discriminador individual do MSD.

    Rede convolucional que classifica se o áudio é real ou sintético
    em uma escala temporal específica.
    """

    def __init__(self, config: VITS2Config, scale: int = 1):
        super().__init__()
        self.config = config
        self.scale = scale

        # Configurações das convoluções
        channels = [1, 16, 64, 256, 1024, 1024, 1024]
        kernel_sizes = [15, 41, 41, 41, 41, 41, 5]
        strides = [1, 2, 2, 4, 4, 1, 1]
        paddings = [7, 20, 20, 20, 20, 20, 2]

        # Camadas convolucionais
        self.convs = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        channels[i],
                        channels[i + 1],
                        kernel_sizes[i],
                        strides[i],
                        paddings[i]
                    ),
                    nn.LeakyReLU(0.1),
                )
            )

        # Camada final
        self.final_conv = nn.Conv1d(channels[-1], 1, kernel_sizes[-1], 1, paddings[-1])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass do sub-discriminador.

        Args:
            x: Input waveform

        Returns:
            Tuple de (output, feature_maps)
        """
        feature_maps = []

        for conv in self.convs:
            x = conv(x)
            feature_maps.append(x)

        # Output final
        output = self.final_conv(x)

        return output, feature_maps


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator (MPD) do VITS2.

    Analisa patterns periódicos no áudio através de
    sub-discriminadores que operam em diferentes períodos.
    """

    def __init__(self, config: VITS2Config):
        super().__init__()
        self.config = config

        # Períodos dos sub-discriminadores
        self.periods = config.mpd_periods

        # Sub-discriminadores para cada período
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(config, period)
            for period in self.periods
        ])

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Forward pass do MPD.

        Args:
            x: Waveform [batch_size, 1, seq_len]

        Returns:
            Tuple de:
                - outputs: Lista de outputs de cada sub-discriminador
                - feature_maps: Lista de feature maps de cada sub-discriminador
        """
        outputs = []
        feature_maps = []

        for discriminator in self.discriminators:
            out, features = discriminator(x)
            outputs.append(out)
            feature_maps.append(features)

        return outputs, feature_maps


class PeriodDiscriminator(nn.Module):
    """
    Sub-discriminador individual do MPD.

    Analisa patterns periódicos no áudio reorganizando
    o waveform em uma estrutura 2D baseada no período.
    """

    def __init__(self, config: VITS2Config, period: int):
        super().__init__()
        self.config = config
        self.period = period

        # Configurações das convoluções 2D
        channels = [1, 32, 128, 512, 1024, 1024]
        kernel_sizes = [(5, 1), (5, 1), (5, 1), (5, 1), (5, 1)]
        strides = [(3, 1), (3, 1), (3, 1), (3, 1), (1, 1)]
        paddings = [(2, 0), (2, 0), (2, 0), (2, 0), (2, 0)]

        # Camadas convolucionais 2D
        self.convs = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels[i],
                        channels[i + 1],
                        kernel_sizes[i],
                        strides[i],
                        paddings[i]
                    ),
                    nn.LeakyReLU(0.1),
                )
            )

        # Camada final
        self.final_conv = nn.Conv2d(channels[-1], 1, (3, 1), 1, (1, 0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass do sub-discriminador.

        Args:
            x: Input waveform [batch_size, 1, seq_len]

        Returns:
            Tuple de (output, feature_maps)
        """
        batch_size, channels, seq_len = x.shape

        # Reorganizar para estrutura 2D baseada no período
        if seq_len % self.period != 0:
            # Pad para tornar divisível pelo período
            n_pad = self.period - (seq_len % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            seq_len = x.shape[-1]

        # Reshape para [batch_size, channels, seq_len // period, period]
        x = x.view(batch_size, channels, seq_len // self.period, self.period)

        feature_maps = []

        # Convoluções 2D
        for conv in self.convs:
            x = conv(x)
            feature_maps.append(x)

        # Output final
        output = self.final_conv(x)

        # Flatten para [batch_size, 1, height * width]
        output = output.flatten(2)

        return output, feature_maps


class DiscriminatorLoss(nn.Module):
    """
    Loss functions para os discriminadores do VITS2.

    Inclui adversarial loss, feature matching loss e
    regularização para estabilidade do treinamento.
    """

    def __init__(self, config: VITS2Config):
        super().__init__()
        self.config = config

        # Pesos das loss functions
        self.adv_loss_weight = config.adversarial_loss_weight
        self.fm_loss_weight = config.feature_matching_loss_weight

    def discriminator_loss(
        self,
        real_outputs: List[torch.Tensor],
        fake_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Calcula loss do discriminador.

        Args:
            real_outputs: Outputs do discriminador para áudio real
            fake_outputs: Outputs do discriminador para áudio sintético

        Returns:
            Loss total do discriminador
        """
        loss = 0.0

        for real_out, fake_out in zip(real_outputs, fake_outputs):
            # Loss para classificar real como real (target = 1)
            real_loss = F.mse_loss(real_out, torch.ones_like(real_out))

            # Loss para classificar fake como fake (target = 0)
            fake_loss = F.mse_loss(fake_out, torch.zeros_like(fake_out))

            loss += real_loss + fake_loss

        return loss

    def generator_loss(
        self,
        fake_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Calcula adversarial loss do generator.

        Args:
            fake_outputs: Outputs do discriminador para áudio sintético

        Returns:
            Adversarial loss do generator
        """
        loss = 0.0

        for fake_out in fake_outputs:
            # Generator tenta enganar discriminador (target = 1)
            loss += F.mse_loss(fake_out, torch.ones_like(fake_out))

        return loss * self.adv_loss_weight

    def feature_matching_loss(
        self,
        real_features: List[List[torch.Tensor]],
        fake_features: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Calcula feature matching loss.

        Compara features intermediárias entre áudio real e sintético
        para melhorar a qualidade perceptual.

        Args:
            real_features: Features do discriminador para áudio real
            fake_features: Features do discriminador para áudio sintético

        Returns:
            Feature matching loss
        """
        loss = 0.0
        num_discriminators = len(real_features)

        for i in range(num_discriminators):
            for real_feat, fake_feat in zip(real_features[i], fake_features[i]):
                loss += F.l1_loss(fake_feat, real_feat.detach())

        return loss * self.fm_loss_weight / num_discriminators


class CombinedDiscriminator(nn.Module):
    """
    Discriminador combinado que une MSD e MPD.

    Fornece interface unificada para ambos os discriminadores
    e calcula losses combinadas.
    """

    def __init__(self, config: VITS2Config):
        super().__init__()
        self.config = config

        # Discriminadores
        self.msd = MultiScaleDiscriminator(config)
        self.mpd = MultiPeriodDiscriminator(config)

        # Loss function
        self.loss_fn = DiscriminatorLoss(config)

    def forward(
        self,
        real_audio: torch.Tensor,
        fake_audio: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass completo dos discriminadores.

        Args:
            real_audio: Áudio real [batch_size, 1, seq_len]
            fake_audio: Áudio sintético [batch_size, 1, seq_len]

        Returns:
            Dict com losses e outputs
        """
        # MSD outputs
        msd_real_outputs, msd_real_features = self.msd(real_audio)
        msd_fake_outputs, msd_fake_features = self.msd(fake_audio.detach())

        # MPD outputs
        mpd_real_outputs, mpd_real_features = self.mpd(real_audio)
        mpd_fake_outputs, mpd_fake_features = self.mpd(fake_audio.detach())

        # Combine outputs
        real_outputs = msd_real_outputs + mpd_real_outputs
        fake_outputs = msd_fake_outputs + mpd_fake_outputs
        real_features = msd_real_features + mpd_real_features
        fake_features = msd_fake_features + mpd_fake_features

        # Calculate losses
        disc_loss = self.loss_fn.discriminator_loss(real_outputs, fake_outputs)

        return {
            'discriminator_loss': disc_loss,
            'real_outputs': real_outputs,
            'fake_outputs': fake_outputs,
            'real_features': real_features,
            'fake_features': fake_features,
        }

    def generator_forward(
        self,
        fake_audio: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass para o generator (sem gradientes no discriminador).

        Args:
            fake_audio: Áudio sintético [batch_size, 1, seq_len]

        Returns:
            Dict com losses do generator
        """
        # MSD outputs (sem detach)
        msd_fake_outputs, msd_fake_features = self.msd(fake_audio)

        # MPD outputs (sem detach)
        mpd_fake_outputs, mpd_fake_features = self.mpd(fake_audio)

        # Combine outputs
        fake_outputs = msd_fake_outputs + mpd_fake_outputs

        # Generator adversarial loss
        gen_adv_loss = self.loss_fn.generator_loss(fake_outputs)

        return {
            'generator_adversarial_loss': gen_adv_loss,
            'fake_outputs': fake_outputs,
            'fake_features': msd_fake_features + mpd_fake_features,
        }
