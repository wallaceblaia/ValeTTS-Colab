"""
Técnicas de data augmentation para áudio e texto.
"""

from valetts.data.augmentation.audio import AudioAugmentation
from valetts.data.augmentation.text import TextAugmentation
from valetts.data.augmentation.prosody import ProsodyAugmentation

__all__ = [
    "AudioAugmentation", 
    "TextAugmentation",
    "ProsodyAugmentation",
] 