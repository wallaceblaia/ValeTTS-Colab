"""
Validador de Segurança para Sistema de Monitoramento LLM.

Garante que as sugestões do LLM estejam dentro de ranges seguros
e não causem instabilidade no treinamento.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from valetts.training.monitoring.config import LLMMonitorConfig

logger = logging.getLogger(__name__)


class SafetyValidator:
    """
    Validador de segurança para parâmetros sugeridos pelo LLM.

    Aplica restrições e limites para garantir que mudanças
    não tornem o treinamento instável ou inviável.
    """

    def __init__(self, config: LLMMonitorConfig):
        """
        Inicializa o validador.

        Args:
            config: Configuração com ranges seguros
        """
        self.config = config
        self.safe_ranges = config.get_safe_ranges()
        self.change_history: List[Dict[str, Any]] = []

    def validate_suggestions(
        self, suggestions: Dict[str, Any], current_values: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Valida e filtra sugestões do LLM.

        Args:
            suggestions: Sugestões do LLM
            current_values: Valores atuais dos parâmetros

        Returns:
            Tuple com (sugestões_validadas, warnings)
        """
        validated = {}
        warnings = []

        for param, suggested_value in suggestions.items():
            if param in current_values:
                current_value = current_values[param]

                # Aplicar validação específica
                valid_value, warning = self._validate_parameter(
                    param, suggested_value, current_value
                )

                if valid_value is not None:
                    validated[param] = valid_value
                    if warning:
                        warnings.append(warning)
                else:
                    warnings.append(
                        f"❌ Parâmetro {param} rejeitado: "
                        f"sugerido={suggested_value}, atual={current_value}"
                    )

        # Verificar limites de mudanças consecutivas
        consecutive_warnings = self._check_consecutive_changes(validated)
        warnings.extend(consecutive_warnings)

        logger.info(
            f"Validação: {len(validated)}/{len(suggestions)} "
            f"sugestões aceitas, {len(warnings)} warnings"
        )

        return validated, warnings

    def _validate_parameter(
        self, param: str, suggested: Any, current: Any
    ) -> Tuple[Optional[Any], Optional[str]]:
        """
        Valida um parâmetro específico.

        Args:
            param: Nome do parâmetro
            suggested: Valor sugerido
            current: Valor atual

        Returns:
            Tuple com (valor_validado, warning)
        """
        # Learning Rate
        if param == "learning_rate":
            return self._validate_learning_rate(suggested, current)

        # Batch Size
        elif param == "batch_size":
            return self._validate_batch_size(suggested, current)

        # Loss Weights
        elif param.startswith("loss_weight_"):
            return self._validate_loss_weight(suggested, current)

        # Scheduler parameters
        elif param.startswith("scheduler_"):
            return self._validate_scheduler_param(param, suggested, current)

        # Parâmetro desconhecido - rejeitar
        else:
            return None, f"Parâmetro desconhecido: {param}"

    def _validate_learning_rate(
        self, suggested: float, current: float
    ) -> Tuple[Optional[float], Optional[str]]:
        """Valida learning rate."""
        ranges = self.safe_ranges["learning_rate"]

        # Verificar range absoluto
        if not (ranges["min"] <= suggested <= ranges["max"]):
            clamped = max(ranges["min"], min(ranges["max"], suggested))
            return clamped, (
                f"⚠️ Learning rate {suggested:.2e} fora do range seguro "
                f"[{ranges['min']:.2e}, {ranges['max']:.2e}]. "
                f"Usando {clamped:.2e}"
            )

        # Verificar mudança abrupta
        if current > 0:
            change_factor = suggested / current
            max_change = ranges["max_change_factor"]

            if change_factor > max_change or change_factor < (1 / max_change):
                # Limitar mudança
                if suggested > current:
                    limited = current * max_change
                else:
                    limited = current / max_change

                return limited, (
                    f"⚠️ Mudança de learning rate muito abrupta "
                    f"({change_factor:.2f}x). Limitando a {max_change}x"
                )

        return suggested, None

    def _validate_batch_size(
        self, suggested: int, current: int
    ) -> Tuple[Optional[int], Optional[str]]:
        """Valida batch size."""
        ranges = self.safe_ranges["batch_size"]

        # Garantir que é inteiro positivo
        suggested = max(1, int(suggested))

        # Verificar range absoluto
        if not (ranges["min"] <= suggested <= ranges["max"]):
            clamped = max(ranges["min"], min(ranges["max"], suggested))
            return clamped, (
                f"⚠️ Batch size {suggested} fora do range seguro "
                f"[{ranges['min']}, {ranges['max']}]. Usando {clamped}"
            )

        # Verificar se é potência de 2 (recomendado para GPU)
        if suggested & (suggested - 1) != 0:
            # Encontrar a potência de 2 mais próxima
            lower_pow2 = 2 ** (suggested.bit_length() - 1)
            upper_pow2 = 2 ** suggested.bit_length()

            if abs(suggested - lower_pow2) <= abs(suggested - upper_pow2):
                adjusted = lower_pow2
            else:
                adjusted = upper_pow2

            # Garantir que está no range
            adjusted = max(ranges["min"], min(ranges["max"], adjusted))

            return adjusted, (
                f"ℹ️ Batch size {suggested} ajustado para {adjusted} "
                f"(potência de 2 para melhor performance GPU)"
            )

        return suggested, None

    def _validate_loss_weight(
        self, suggested: float, current: float
    ) -> Tuple[Optional[float], Optional[str]]:
        """Valida pesos de loss."""
        ranges = self.safe_ranges["loss_weights"]

        # Verificar range absoluto
        if not (ranges["min"] <= suggested <= ranges["max"]):
            clamped = max(ranges["min"], min(ranges["max"], suggested))
            return clamped, (
                f"⚠️ Loss weight {suggested:.3f} fora do range seguro. "
                f"Usando {clamped:.3f}"
            )

        # Verificar mudança gradual
        if current > 0:
            change_factor = suggested / current
            max_change = ranges["max_change_factor"]

            if change_factor > max_change or change_factor < (1 / max_change):
                if suggested > current:
                    limited = current * max_change
                else:
                    limited = current / max_change

                return limited, (
                    f"⚠️ Mudança de loss weight muito abrupta. "
                    f"Limitando mudança a {max_change}x"
                )

        return suggested, None

    def _validate_scheduler_param(
        self, param: str, suggested: Any, current: Any
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Valida parâmetros do scheduler."""
        if "warmup" in param:
            ranges = self.safe_ranges["scheduler_warmup"]
            suggested = max(ranges["min"], min(ranges["max"], int(suggested)))

        elif "patience" in param:
            ranges = self.safe_ranges["scheduler_patience"]
            suggested = max(ranges["min"], min(ranges["max"], int(suggested)))

        return suggested, None

    def _check_consecutive_changes(self, validated: Dict[str, Any]) -> List[str]:
        """
        Verifica se há muitas mudanças consecutivas.

        Args:
            validated: Parâmetros validados

        Returns:
            Lista de warnings sobre mudanças consecutivas
        """
        warnings = []

        # Contar mudanças recentes
        recent_changes = len(
            [
                h
                for h in self.change_history[-self.config.max_consecutive_changes :]
                if any(param in validated for param in h.get("params", []))
            ]
        )

        if recent_changes >= self.config.max_consecutive_changes:
            warnings.append(
                f"⚠️ Muitas mudanças consecutivas ({recent_changes}). "
                f"Considere aguardar mais épocas para estabilização."
            )

        return warnings

    def record_change(self, epoch: int, params: Dict[str, Any]) -> None:
        """
        Registra uma mudança no histórico.

        Args:
            epoch: Época da mudança
            params: Parâmetros alterados
        """
        self.change_history.append(
            {"epoch": epoch, "params": list(params.keys()), "values": params.copy()}
        )

        # Manter apenas histórico recente
        if len(self.change_history) > self.config.max_history_entries:
            self.change_history = self.change_history[
                -self.config.max_history_entries :
            ]

    def is_critical_change_batch(
        self, suggestions: Dict[str, Any], current: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Verifica se o conjunto de mudanças é crítico.

        Args:
            suggestions: Sugestões validadas
            current: Valores atuais

        Returns:
            Tuple com (é_crítico, lista_de_mudanças_críticas)
        """
        critical_changes = []

        for param, new_value in suggestions.items():
            if param in current:
                old_value = current[param]
                if self.config.is_critical_change(param, old_value, new_value):
                    critical_changes.append(f"{param}: {old_value} → {new_value}")

        is_critical = len(critical_changes) > 0
        return is_critical, critical_changes
