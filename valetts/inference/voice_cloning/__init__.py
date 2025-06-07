"""
Sistema de clonagem de voz few-shot.
"""

from valetts.inference.voice_cloning.cloner import VoiceCloner
from valetts.inference.voice_cloning.enrollment import SpeakerEnrollment
from valetts.inference.voice_cloning.adaptation import FastAdaptation

__all__ = [
    "VoiceCloner",
    "SpeakerEnrollment",
    "FastAdaptation",
] 