"""
Implementação da arquitetura VITS2 para ValeTTS.

VITS2 é um modelo end-to-end que sintetiza diretamente texto para forma de onda
usando Variational Autoencoders com treinamento adversarial.
"""

from .config import VITS2Config
from .encoder import TextEncoder, PosteriorEncoder
from .decoder import Generator
from .discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from .model import VITS2

__all__ = [
    "VITS2",
    "VITS2Config",
    "TextEncoder",
    "PosteriorEncoder",
    "Generator",
    "MultiPeriodDiscriminator",
    "MultiScaleDiscriminator",
]
