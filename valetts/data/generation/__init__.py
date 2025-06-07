"""
Sistema de Geração de Datasets TTS
=================================

Módulo responsável pela geração automática de datasets para treinamento TTS
usando Edge-TTS e outras fontes de texto e áudio.
"""

from .audio_processor import AudioProcessor
from .dataset_builder import DEFAULT_CONFIG, DatasetBuilder
from .edge_tts_interface import EdgeTTSInterface
from .text_generator import TextGenerator

__all__ = [
    "TextGenerator",
    "EdgeTTSInterface",
    "AudioProcessor",
    "DatasetBuilder",
    "DEFAULT_CONFIG",
]
