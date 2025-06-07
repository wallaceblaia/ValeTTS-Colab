"""
Módulo de controle prosódico avançado.

Implementa Global Style Tokens (GST), Style Adaptive Layer Normalization (SALN)
e DrawSpeech para controle fino de prosódia.
"""

from valetts.models.prosody.controls import ProsodyControls
from valetts.models.prosody.drawspeech import DrawSpeech
from valetts.models.prosody.gst import GlobalStyleTokens
from valetts.models.prosody.saln import StyleAdaptiveLayerNorm

__all__ = [
    "GlobalStyleTokens",
    "StyleAdaptiveLayerNorm",
    "DrawSpeech",
    "ProsodyControls",
]
