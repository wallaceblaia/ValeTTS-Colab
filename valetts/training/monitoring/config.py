"""
Configuração para Sistema de Monitoramento Assistido por LLM.

Define ranges seguros e parâmetros configuráveis para análise
inteligente do treinamento por modelos de linguagem.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMMonitorConfig:
    """
    Configuração para monitoramento de treinamento assistido por LLM.

    Inclui ranges seguros para parâmetros que o LLM pode sugerir
    e configurações de operação do sistema.
    """

    # === Configuração do LLM ===
    enabled: bool = False
    provider: str = "openrouter"
    model: str = "anthropic/claude-3-5-sonnet-20241022"
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"

    # === Intervalos de Monitoramento ===
    monitor_every_epochs: int = 5
    monitor_every_steps: Optional[int] = None  # Se None, usa epochs
    save_analysis_history: bool = True
    max_history_entries: int = 100

    # === Relatórios Informativos ===
    generate_reports: bool = False
    report_every_epochs: int = 5

    # === Arquivo de Configuração Dinâmica ===
    dynamic_config_path: str = "configs/training/llm_dynamic_config.yaml"
    backup_original_config: bool = True

    # === Ranges Seguros para Learning Rate ===
    lr_min: float = 1e-6
    lr_max: float = 1e-2
    lr_change_factor_max: float = 2.0  # Máximo 2x mudança por vez

    # === Ranges Seguros para Batch Size ===
    batch_size_min: int = 4
    batch_size_max: int = 128
    batch_size_change_factor_max: float = 2.0

    # === Ranges Seguros para Loss Weights ===
    loss_weight_min: float = 0.1
    loss_weight_max: float = 10.0
    loss_weight_change_factor_max: float = 1.5

    # === Ranges Seguros para Scheduler ===
    scheduler_warmup_min: int = 100
    scheduler_warmup_max: int = 5000
    scheduler_patience_min: int = 3
    scheduler_patience_max: int = 20

    # === Configuração de Segurança ===
    require_human_approval: bool = True  # Para mudanças críticas
    critical_change_threshold: float = 0.3  # 30% mudança = crítica
    max_consecutive_changes: int = 3  # Máx mudanças consecutivas

    # === Métricas Monitoradas ===
    metrics_to_monitor: List[str] = field(
        default_factory=lambda: [
            "train_loss_total",
            "train_loss_mel",
            "train_loss_kl",
            "train_loss_adv",
            "val_loss_total",
            "val_loss_mel",
            "learning_rate",
            "grad_norm",
            "epoch_time",
            "memory_usage",
        ]
    )

    # === Limites de Alerta ===
    loss_spike_threshold: float = 2.0  # 2x aumento = alerta
    grad_norm_threshold: float = 10.0
    memory_usage_threshold: float = 0.9  # 90% da VRAM

    # === Configuração de Prompt ===
    prompt_template_path: str = "configs/training/llm_analysis_prompt.txt"
    include_context_epochs: int = 3  # Épocas anteriores para contexto

    def __post_init__(self):
        """Validação pós-inicialização."""
        # Verificar se API key existe
        if self.enabled and not os.getenv(self.api_key_env):
            raise ValueError(f"API key não encontrada: {self.api_key_env}")

        # Validar ranges
        if self.lr_min >= self.lr_max:
            raise ValueError("lr_min deve ser menor que lr_max")

        if self.batch_size_min >= self.batch_size_max:
            raise ValueError(
                "batch_size_min deve ser menor que batch_size_max"
            )

    def get_safe_ranges(self) -> Dict[str, Dict[str, float]]:
        """
        Retorna dicionário com ranges seguros para o LLM.

        Returns:
            Dict com ranges para cada parâmetro
        """
        return {
            "learning_rate": {
                "min": self.lr_min,
                "max": self.lr_max,
                "max_change_factor": self.lr_change_factor_max,
            },
            "batch_size": {
                "min": self.batch_size_min,
                "max": self.batch_size_max,
                "max_change_factor": self.batch_size_change_factor_max,
            },
            "loss_weights": {
                "min": self.loss_weight_min,
                "max": self.loss_weight_max,
                "max_change_factor": self.loss_weight_change_factor_max,
            },
            "scheduler_warmup": {
                "min": self.scheduler_warmup_min,
                "max": self.scheduler_warmup_max,
            },
            "scheduler_patience": {
                "min": self.scheduler_patience_min,
                "max": self.scheduler_patience_max,
            },
        }

    def is_critical_change(
        self, param_name: str, old_value: float, new_value: float
    ) -> bool:
        """
        Verifica se uma mudança é considerada crítica.

        Args:
            param_name: Nome do parâmetro
            old_value: Valor atual
            new_value: Valor sugerido

        Returns:
            True se a mudança for crítica
        """
        if old_value == 0:
            return abs(new_value) > self.critical_change_threshold

        change_ratio = abs(new_value - old_value) / abs(old_value)
        return change_ratio > self.critical_change_threshold


@dataclass
class LLMAnalysisResponse:
    """
    Estrutura da resposta do LLM sobre análise do treinamento.
    """

    # === Análise Geral ===
    overall_status: str  # "healthy", "warning", "critical"
    confidence_score: float  # 0.0 a 1.0
    analysis_summary: str

    # === Sugestões de Configuração ===
    config_suggestions: Dict[str, Any]

    # === Observações Específicas ===
    observations: List[str]

    # === Medidas Recomendadas ===
    recommended_actions: List[str]

    # === Flags de Alerta ===
    alerts: List[str]

    # === Próximos Passos ===
    next_checkpoints: List[str]

    # === Metadados ===
    timestamp: str
    epoch: int
    model_used: str
