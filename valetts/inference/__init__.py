"""
Sistema de inferência para síntese de fala e clonagem de voz.

Fornece interfaces de alto nível para síntese de fala,
clonagem de voz e APIs de produção.
"""

from valetts.inference.api import TTSServer
from valetts.inference.synthesizers import TTSSynthesizer
from valetts.inference.voice_cloning import VoiceCloner

__all__ = [
    "TTSSynthesizer",
    "VoiceCloner",
    "TTSServer",
]
