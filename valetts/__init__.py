"""
ValeTTS - Sistema TTS de Última Geração com Arquitetura Híbrida

Sistema avançado de síntese de fala que combina:
- VITS2 para síntese single-stage
- Meta-Learning (MAML) para few-shot learning
- BigVGAN-v2 para vocoder de alta performance
- Controle prosódico avançado com GST e SALN
- Suporte multilíngue com zero-shot cross-lingual

Exemplos básicos:
    >>> from valetts import TTSSynthesizer
    >>> synthesizer = TTSSynthesizer()
    >>> audio = synthesizer.synthesize("Hello world", language="en")
    
    >>> from valetts import VoiceCloner
    >>> cloner = VoiceCloner()
    >>> cloned_audio = cloner.clone_voice("Hello", reference_audio="speaker.wav")
"""

__version__ = "0.1.0"
__author__ = "Equipe ValeTTS"
__email__ = "contato@valetts.dev"

# Core exports
from valetts.inference.synthesizers import TTSSynthesizer
from valetts.inference.voice_cloning import VoiceCloner
from valetts.models.vits2 import VITS2
from valetts.utils.audio import AudioProcessor
from valetts.utils.text import TextProcessor

__all__ = [
    "TTSSynthesizer",
    "VoiceCloner", 
    "VITS2",
    "AudioProcessor",
    "TextProcessor",
    "__version__",
] 