"""
Implementação de Meta-Learning (MAML) para few-shot voice cloning.

Model-Agnostic Meta-Learning permite adaptação rápida a novos speakers
com apenas 5-6 segundos de áudio de referência.
"""

from valetts.models.meta_learning.episode import EpisodeSampler
from valetts.models.meta_learning.few_shot import FewShotLearner
from valetts.models.meta_learning.maml import MAML

__all__ = [
    "MAML",
    "FewShotLearner",
    "EpisodeSampler",
]
