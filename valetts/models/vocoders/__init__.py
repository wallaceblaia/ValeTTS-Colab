"""
Módulo de vocoders neurais para síntese de áudio.

Implementa BigVGAN-v2 e outros vocoders estado da arte para
geração de áudio de alta qualidade.
"""

from valetts.models.vocoders.base import BaseVocoder
from valetts.models.vocoders.bigvgan import BigVGAN_v2
from valetts.models.vocoders.hifigan import HiFiGAN

__all__ = [
    "BigVGAN_v2",
    "HiFiGAN",
    "BaseVocoder",
]
