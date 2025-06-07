"""
Módulo de modelos do ValeTTS.

Contém todas as arquiteturas de modelos implementadas:
- VITS2: Arquitetura base end-to-end
- Meta-Learning: MAML para few-shot learning
- Prosody: Controles prosódicos (GST, SALN, DrawSpeech)
- Vocoders: BigVGAN-v2 e outros vocoders neurais
"""

from valetts.models.base import BaseModel, BaseModelConfig
from valetts.models.vits2 import VITS2

__all__ = [
    "VITS2",
    "BaseModel",
    "BaseModelConfig",
]
