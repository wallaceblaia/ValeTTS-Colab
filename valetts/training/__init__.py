"""
Sistema de treinamento para modelos ValeTTS.

Inclui trainers especializados, otimizadores, schedulers
e monitoramento assistido por LLM para treinamento distribuído eficiente.
"""

# from valetts.training.optimizers import CustomOptimizer
# from valetts.training.schedulers import WarmupScheduler
# from valetts.training.trainers import MAMLTrainer, VITS2Trainer

# Sistema de Monitoramento LLM - Funcionalidade inovadora
from valetts.training.monitoring import (
    LLMMonitorConfig,
    LLMTrainingMonitor,
    SafetyValidator,
)

__all__ = [
    # "VITS2Trainer",
    # "MAMLTrainer",
    # "CustomOptimizer",
    # "WarmupScheduler",
    # Sistema de Monitoramento LLM
    "LLMTrainingMonitor",
    "LLMMonitorConfig",
    "SafetyValidator",
]
