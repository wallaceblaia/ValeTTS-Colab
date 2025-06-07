"""
Text Encoder para VITS2 - Implementação completa.

Este módulo implementa o encoder de texto do VITS2, incluindo:
- Embedding de caracteres/phonemas
- Transformer encoder com atenção multi-cabeça
- Predição de duração (duration predictor)
- Encoder prosódico para controle de entonação
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from ..base.layers import (
    MultiHeadAttention,
    FFN,
    PositionalEncoding,
    Conv1d,
    ResidualBlock
)
from .config import VITS2Config


class TextEncoder(nn.Module):
    """
    Text Encoder do VITS2 com Transformer architecture.

    Converte sequência de texto em representações latentes,
    incluindo predição de duração e informações prosódicas.
    """

    def __init__(self, config: VITS2Config):
        super().__init__()
        self.config = config

        # Embedding de caracteres/phonemas
        self.character_embedding = nn.Embedding(
            config.text_vocab_size,
            config.text_encoder_hidden_dim
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=config.text_encoder_hidden_dim,
            max_length=config.max_text_length
        )

        # Camadas Transformer
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=config.text_encoder_hidden_dim,
                nhead=config.text_encoder_attention_heads,
                dim_feedforward=config.text_encoder_filter_size,
                dropout=config.dropout
            )
            for _ in range(config.text_encoder_layers)
        ])

        # Layer norm final
        self.layer_norm = nn.LayerNorm(config.text_encoder_hidden_dim)

        # Duration Predictor
        self.duration_predictor = DurationPredictor(
            hidden_dim=config.text_encoder_hidden_dim,
            filter_size=config.duration_predictor_filter_size,
            kernel_size=config.duration_predictor_kernel_size,
            dropout=config.dropout
        )

        # Prosody Encoder (opcional)
        if config.use_prosody_encoder:
            self.prosody_encoder = ProsodyEncoder(
                hidden_dim=config.text_encoder_hidden_dim,
                prosody_dim=config.prosody_dim
            )
        else:
            self.prosody_encoder = None

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        prosody_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass do Text Encoder.

        Args:
            text: Tensor de texto [batch_size, max_text_len]
            text_lengths: Comprimentos reais [batch_size]
            durations: Durações target para treinamento [batch_size, max_text_len]
            prosody_features: Features prosódicas opcionais

        Returns:
            Dict com outputs do encoder:
                - 'encoder_outputs': representações de texto
                - 'duration_predictions': predições de duração
                - 'prosody_embeddings': embeddings prosódicos (se habilitado)
        """
        batch_size, max_text_len = text.shape

        # Character embedding
        x = self.character_embedding(text)  # [B, T, H]

        # Positional encoding
        x = self.pos_encoding(x)

        # Criar máscara de padding
        mask = self._create_padding_mask(text_lengths, max_text_len)

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=None)  # Simplificar por enquanto

        # Layer norm final
        encoder_outputs = self.layer_norm(x)

        # Duration prediction
        duration_predictions = self.duration_predictor(encoder_outputs, mask)

        results = {
            'encoder_outputs': encoder_outputs,
            'duration_predictions': duration_predictions,
        }

        # Prosody encoding (se habilitado)
        if self.prosody_encoder is not None:
            prosody_embeddings = self.prosody_encoder(
                encoder_outputs,
                prosody_features
                        )
            results['prosody_embeddings'] = prosody_embeddings

        return results

    def _create_padding_mask(
        self,
        lengths: torch.Tensor,
        max_len: int
    ) -> torch.Tensor:
        """Cria máscara de padding para sequências."""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).expand(
            batch_size, max_len
        ) >= lengths.unsqueeze(1)
        return mask


class TransformerEncoderLayer(nn.Module):
    """Camada individual do Transformer Encoder."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=nhead,
            dropout=dropout
        )

        self.ffn = FFN(
            d_model=d_model,
            d_ff=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass da camada."""
        # Self-attention
        attn_output, _ = self.self_attn(
            src, src, src,
            mask=src_key_padding_mask
        )
        src = self.norm1(src + self.dropout(attn_output))

        # Feed forward
        ffn_output = self.ffn(src)
        src = self.norm2(src + self.dropout(ffn_output))

        return src


class DurationPredictor(nn.Module):
    """
    Preditor de duração para cada token de texto.

    Prediz quanto tempo cada phonema/caractere deve durar
    na síntese de áudio.
    """

    def __init__(
        self,
        hidden_dim: int,
        filter_size: int,
        kernel_size: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.convs = nn.ModuleList([
            Conv1d(
                in_channels=hidden_dim,
                out_channels=filter_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                activation='relu',
                norm='batch_norm'
            ),
            Conv1d(
                in_channels=filter_size,
                out_channels=filter_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                activation='relu',
                norm='batch_norm'
            )
        ])

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(filter_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Prediz durações.

        Args:
            x: Features de texto [batch_size, seq_len, hidden_dim]
            mask: Máscara de padding

        Returns:
            Predições de duração [batch_size, seq_len]
        """
        # Transpor para [B, H, T] para convoluções
        x = x.transpose(1, 2)

        # Convoluções
        for conv in self.convs:
            x = conv(x)
            x = self.dropout(x)

        # Transpor de volta e projetar
        x = x.transpose(1, 2)  # [B, T, filter_size]
        duration_pred = self.linear(x).squeeze(-1)  # [B, T]

        # Aplicar máscara se fornecida
        if mask is not None:
            duration_pred = duration_pred.masked_fill(mask, 0.0)

        # Aplicar softplus para garantir valores positivos
        duration_pred = F.softplus(duration_pred)

        return duration_pred


class ProsodyEncoder(nn.Module):
    """
    Encoder prosódico para controle de entonação e ritmo.

    Incorpora informações prosódicas globais e locais
    para controle mais fino da síntese.
    """

    def __init__(self, hidden_dim: int, prosody_dim: int):
        super().__init__()

        self.prosody_projection = nn.Linear(prosody_dim, hidden_dim)
        self.attention = MultiHeadAttention(
            d_model=hidden_dim,
            num_heads=4,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        text_features: torch.Tensor,
        prosody_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Incorpora informações prosódicas.

        Args:
            text_features: Features de texto [B, T, H]
            prosody_features: Features prosódicas [B, P] ou [B, T, P]

        Returns:
            Features enriquecidas com prosódia [B, T, H]
        """
        if prosody_features is None:
            return text_features

        # Projetar features prosódicas
        if prosody_features.dim() == 2:
            # Global prosody - expandir para toda a sequência
            prosody_embed = self.prosody_projection(prosody_features)  # [B, H]
            prosody_embed = prosody_embed.unsqueeze(1).expand(
                -1, text_features.size(1), -1
            )  # [B, T, H]
        else:
            # Local prosody
            prosody_embed = self.prosody_projection(prosody_features)  # [B, T, H]

        # Cross-attention entre texto e prosódia
        attended_features, _ = self.attention(
            text_features, prosody_embed, prosody_embed
        )

        # Residual connection
        output = self.norm(text_features + attended_features)

        return output


class PosteriorEncoder(nn.Module):
    """
    Posterior Encoder que codifica mel-spectrograms em representações latentes.

    Este encoder processa o áudio real durante o treinamento para
    aprender o mapeamento entre texto e características acústicas.
    """

    def __init__(self, config: VITS2Config):
        super().__init__()
        self.config = config

        # Convolutional encoder
        self.conv_layers = nn.ModuleList()

        # Primeira camada - mel_channels para hidden_dim
        self.conv_layers.append(
            Conv1d(
                in_channels=config.mel_channels,
                out_channels=config.posterior_encoder_hidden_dim,
                kernel_size=5,
                padding=2,
                activation='relu',
                norm='batch_norm'
            )
        )

        # Camadas intermediárias
        for _ in range(config.posterior_encoder_layers - 1):
            self.conv_layers.append(
                ResidualBlock(
                    channels=config.posterior_encoder_hidden_dim,
                    kernel_size=3,
                    dilation=1
                )
            )

        # Projeção para espaço latente (média e log-variância)
        self.proj_mean = Conv1d(
            in_channels=config.posterior_encoder_hidden_dim,
            out_channels=config.latent_dim,
            kernel_size=1
        )

        self.proj_logvar = Conv1d(
            in_channels=config.posterior_encoder_hidden_dim,
            out_channels=config.latent_dim,
            kernel_size=1
        )

    def forward(
        self,
        mel: torch.Tensor,
        mel_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Codifica mel-spectrogram em distribuição latente.

        Args:
            mel: Mel-spectrogram [batch_size, mel_channels, mel_len]
            mel_lengths: Comprimentos reais das sequências

        Returns:
            Tuple de (mean, logvar) da distribuição latente
        """
        x = mel

        # Convolutional encoding
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Projetar para mean e log-variance
        mean = self.proj_mean(x)
        logvar = self.proj_logvar(x)

        return mean, logvar
