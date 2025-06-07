"""
Trainers especializados para diferentes modelos e tarefas.
"""

from valetts.training.trainers.base import BaseTrainer
from valetts.training.trainers.vits2 import VITS2Trainer

__all__ = [
    "VITS2Trainer",
    "BaseTrainer",
]
