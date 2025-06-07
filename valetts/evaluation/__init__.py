"""
Sistema de avaliação e métricas para ValeTTS.

Implementa métricas automatizadas, benchmarking e ferramentas
de análise de qualidade.
"""

from valetts.evaluation.metrics import MOSPredictor, SpeakerSimilarity, RTFBenchmark
from valetts.evaluation.benchmarks import QualityBenchmark, PerformanceBenchmark

__all__ = [
    "MOSPredictor",
    "SpeakerSimilarity", 
    "RTFBenchmark",
    "QualityBenchmark",
    "PerformanceBenchmark",
] 