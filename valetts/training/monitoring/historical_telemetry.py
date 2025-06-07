# Sistema de Telemetria Histórica para LLM
# Coleta e organiza dados históricos para análise mais rica

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """Métricas de uma época específica."""

    epoch: int

    # Training metrics
    train_loss_total: float
    train_loss_mel: float
    train_loss_kl: float
    train_loss_adv: float
    train_loss_fm: float
    train_loss_duration: float

    # Validation metrics
    val_loss_total: float
    val_loss_mel: float
    val_loss_kl: float

    # Training info
    learning_rate: float
    grad_norm: float
    epoch_time: float
    memory_usage: float

    # Optional model-specific metrics
    extra_metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        result = asdict(self)
        if self.extra_metrics is None:
            result.pop("extra_metrics")
        else:
            result.update(self.extra_metrics)
        return result


class HistoricalTelemetryCollector:
    """Coleta telemetria histórica para análise LLM aprimorada."""

    def __init__(self, max_history_size: int = 100):
        self.max_history_size = max_history_size
        self.epochs_history: List[EpochMetrics] = []

    def add_epoch_metrics(self, metrics: Dict[str, float], epoch: int) -> None:
        """Adiciona métricas de uma época."""
        try:
            # Extrair métricas principais
            epoch_data = EpochMetrics(
                epoch=epoch,
                # Training losses
                train_loss_total=metrics.get("train_loss_total", 0.0),
                train_loss_mel=metrics.get("train_loss_mel", 0.0),
                train_loss_kl=metrics.get("train_loss_kl", 0.0),
                train_loss_adv=metrics.get("train_loss_adv", 0.0),
                train_loss_fm=metrics.get("train_loss_fm", 0.0),
                train_loss_duration=metrics.get("train_loss_duration", 0.0),
                # Validation losses
                val_loss_total=metrics.get("val_loss_total", 0.0),
                val_loss_mel=metrics.get("val_loss_mel", 0.0),
                val_loss_kl=metrics.get("val_loss_kl", 0.0),
                # Training info
                learning_rate=metrics.get("learning_rate", 0.0),
                grad_norm=metrics.get("grad_norm", 0.0),
                epoch_time=metrics.get("epoch_time", 0.0),
                memory_usage=metrics.get("memory_usage", 0.0),
                # Métricas extras
                extra_metrics={
                    k: v
                    for k, v in metrics.items()
                    if k
                    not in [
                        "train_loss_total",
                        "train_loss_mel",
                        "train_loss_kl",
                        "train_loss_adv",
                        "train_loss_fm",
                        "train_loss_duration",
                        "val_loss_total",
                        "val_loss_mel",
                        "val_loss_kl",
                        "learning_rate",
                        "grad_norm",
                        "epoch_time",
                        "memory_usage",
                    ]
                },
            )

            self.epochs_history.append(epoch_data)

            # Manter histórico limitado
            if len(self.epochs_history) > self.max_history_size:
                self.epochs_history = self.epochs_history[-self.max_history_size :]

            logger.debug(f"📊 Adicionadas métricas da época {epoch} ao histórico")

        except Exception as e:
            logger.error(f"❌ Erro adicionando métricas ao histórico: {e}")

    def get_contextual_data(self, current_epoch: int) -> Dict[str, Any]:
        """
        Gera dados contextuais inteligentes para o LLM.

        Estratégia:
        - Primeiras 10 épocas (se disponíveis)
        - Últimas 20 épocas (se disponíveis)
        - Época atual sempre incluída
        """
        if not self.epochs_history:
            return {"epochs": [], "summary": "Sem histórico disponível"}

        # Filtrar épocas disponíveis
        available_epochs = [e for e in self.epochs_history if e.epoch <= current_epoch]

        if not available_epochs:
            return {"epochs": [], "summary": "Sem épocas válidas"}

        # Estratégia de seleção inteligente
        selected_epochs = self._select_representative_epochs(
            available_epochs, current_epoch
        )

        # Gerar resumo estatístico
        summary = self._generate_summary_statistics(available_epochs)

        # Análise de tendências
        trends = self._analyze_trends(available_epochs)

        return {
            "epochs": [epoch.to_dict() for epoch in selected_epochs],
            "summary": summary,
            "trends": trends,
            "total_epochs_available": len(available_epochs),
            "selection_strategy": self._get_selection_strategy_description(
                len(available_epochs)
            ),
        }

    def _select_representative_epochs(
        self, available_epochs: List[EpochMetrics], current_epoch: int
    ) -> List[EpochMetrics]:
        """Seleciona épocas representativas baseado na estratégia inteligente."""

        if len(available_epochs) <= 25:
            # Se temos poucas épocas, enviar todas
            return available_epochs

        selected = set()
        result = []

        # 1. Primeiras 10 épocas (se disponíveis)
        first_epochs = [e for e in available_epochs if e.epoch < 10]
        for epoch in first_epochs[:10]:
            if epoch.epoch not in selected:
                selected.add(epoch.epoch)
                result.append(epoch)

        # 2. Últimas 20 épocas
        last_epochs = available_epochs[-20:]
        for epoch in last_epochs:
            if epoch.epoch not in selected:
                selected.add(epoch.epoch)
                result.append(epoch)

        # 3. Garantir que época atual está incluída
        current_epoch_data = next(
            (e for e in available_epochs if e.epoch == current_epoch), None
        )
        if current_epoch_data and current_epoch not in selected:
            result.append(current_epoch_data)

        # Ordenar por época
        result.sort(key=lambda x: x.epoch)

        return result

    def _generate_summary_statistics(
        self, epochs: List[EpochMetrics]
    ) -> Dict[str, Any]:
        """Gera estatísticas resumidas do histórico."""
        if not epochs:
            return {}

        # Coletar métricas principais
        train_losses = [e.train_loss_total for e in epochs]
        val_losses = [e.val_loss_total for e in epochs]
        learning_rates = [e.learning_rate for e in epochs]
        grad_norms = [e.grad_norm for e in epochs]

        return {
            "train_loss": {
                "min": min(train_losses),
                "max": max(train_losses),
                "mean": sum(train_losses) / len(train_losses),
                "latest": train_losses[-1],
            },
            "val_loss": {
                "min": min(val_losses),
                "max": max(val_losses),
                "mean": sum(val_losses) / len(val_losses),
                "latest": val_losses[-1],
            },
            "learning_rate": {
                "min": min(learning_rates),
                "max": max(learning_rates),
                "current": learning_rates[-1],
            },
            "grad_norm": {
                "min": min(grad_norms),
                "max": max(grad_norms),
                "mean": sum(grad_norms) / len(grad_norms),
                "latest": grad_norms[-1],
            },
            "total_epochs": len(epochs),
            "epoch_range": f"{epochs[0].epoch}-{epochs[-1].epoch}",
        }

    def _analyze_trends(self, epochs: List[EpochMetrics]) -> Dict[str, Any]:
        """Analisa tendências nas métricas."""
        if len(epochs) < 3:
            return {
                "trend_analysis": "Histórico insuficiente para análise de tendências"
            }

        # Análise das últimas N épocas
        recent_epochs = epochs[-5:] if len(epochs) >= 5 else epochs[-3:]

        trends = {}

        # Tendência de loss
        train_losses = [e.train_loss_total for e in recent_epochs]
        val_losses = [e.val_loss_total for e in recent_epochs]

        trends["train_loss_trend"] = self._calculate_trend(train_losses)
        trends["val_loss_trend"] = self._calculate_trend(val_losses)

        # Análise de convergência
        if len(epochs) >= 10:
            trends["convergence_analysis"] = self._analyze_convergence(epochs)

        # Detecção de problemas
        trends["potential_issues"] = self._detect_potential_issues(epochs)

        return trends

    def _calculate_trend(self, values: List[float]) -> str:
        """Calcula tendência de uma série de valores."""
        if len(values) < 2:
            return "stable"

        # Calcular diferença média
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        avg_diff = sum(diffs) / len(diffs)

        # Threshold baseado no valor médio
        avg_value = sum(values) / len(values)
        threshold = avg_value * 0.01  # 1% do valor médio

        if avg_diff < -threshold:
            return "decreasing"  # Melhorando
        elif avg_diff > threshold:
            return "increasing"  # Piorando
        else:
            return "stable"

    def _analyze_convergence(self, epochs: List[EpochMetrics]) -> Dict[str, Any]:
        """Analisa sinais de convergência."""
        # Últimas 10 épocas
        recent = epochs[-10:]

        train_losses = [e.train_loss_total for e in recent]
        val_losses = [e.val_loss_total for e in recent]

        # Calcular variância
        train_variance = self._calculate_variance(train_losses)
        val_variance = self._calculate_variance(val_losses)

        # Analisar estagnação
        train_stagnation = max(train_losses) - min(train_losses) < 0.01
        val_stagnation = max(val_losses) - min(val_losses) < 0.01

        return {
            "train_stability": (
                "high"
                if train_variance < 0.01
                else "medium" if train_variance < 0.1 else "low"
            ),
            "val_stability": (
                "high"
                if val_variance < 0.01
                else "medium" if val_variance < 0.1 else "low"
            ),
            "possible_stagnation": train_stagnation and val_stagnation,
            "train_variance": train_variance,
            "val_variance": val_variance,
        }

    def _calculate_variance(self, values: List[float]) -> float:
        """Calcula variância de uma lista de valores."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

    def _detect_potential_issues(self, epochs: List[EpochMetrics]) -> List[str]:
        """Detecta potenciais problemas no treinamento."""
        issues = []

        if len(epochs) < 3:
            return issues

        latest = epochs[-1]

        # KL Loss infinita
        if latest.train_loss_kl == float("inf") or latest.val_loss_kl == float("inf"):
            issues.append("KL divergence infinita - possível instabilidade no VAE")

        # Gradient norm muito alto
        if latest.grad_norm > 10.0:
            issues.append("Norma do gradiente alta - possível gradient explosion")

        # Loss aumentando consistentemente
        recent_train = [e.train_loss_total for e in epochs[-5:]]
        if len(recent_train) >= 3 and all(
            recent_train[i] < recent_train[i + 1] for i in range(len(recent_train) - 1)
        ):
            issues.append("Training loss aumentando consistentemente")

        # Overfitting
        if latest.val_loss_total > latest.train_loss_total * 1.5:
            issues.append(
                "Possível overfitting - validation loss muito maior que training loss"
            )

        return issues

    def _get_selection_strategy_description(self, total_epochs: int) -> str:
        """Descreve a estratégia de seleção usada."""
        if total_epochs <= 25:
            return f"Enviando todas as {total_epochs} épocas disponíveis"
        else:
            return f"Estratégia inteligente: primeiras 10 + últimas 20 épocas (de {total_epochs} total)"
