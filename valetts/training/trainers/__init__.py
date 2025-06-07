"""
Trainers especializados para diferentes modelos e tarefas.
"""

from valetts.training.trainers.vits2 import VITS2Trainer
from valetts.training.trainers.maml import MAMLTrainer  
from valetts.training.trainers.base import BaseTrainer

__all__ = [
    "VITS2Trainer",
    "MAMLTrainer",
    "BaseTrainer",
] 