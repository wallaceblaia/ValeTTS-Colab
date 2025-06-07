"""
Engines de síntese de fala.
"""

from valetts.inference.synthesizers.tts import TTSSynthesizer
from valetts.inference.synthesizers.multilingual import MultilingualSynthesizer
from valetts.inference.synthesizers.base import BaseSynthesizer

__all__ = [
    "TTSSynthesizer",
    "MultilingualSynthesizer", 
    "BaseSynthesizer",
] 