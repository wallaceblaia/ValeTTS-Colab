"""
Sistema de avaliação e métricas para ValeTTS.

Implementa métricas automatizadas, benchmarking e ferramentas
de análise de qualidade.
"""

from valetts.evaluation.benchmarks import PerformanceBenchmark, QualityBenchmark
from valetts.evaluation.metrics import MOSPredictor, RTFBenchmark, SpeakerSimilarity

__all__ = [
    "MOSPredictor",
    "SpeakerSimilarity",
    "RTFBenchmark",
    "QualityBenchmark",
    "PerformanceBenchmark",
]
