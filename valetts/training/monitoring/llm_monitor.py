"""
Monitor de Treinamento Assistido por LLM.

Sistema principal que coordena análise inteligente do progresso
de treinamento usando Claude 4 Sonnet via OpenRoutes.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
import torch
import yaml

from valetts.training.monitoring.config import (
    LLMAnalysisResponse,
    LLMMonitorConfig,
)
from valetts.training.monitoring.historical_telemetry import (
    HistoricalTelemetryCollector,
)
from valetts.training.monitoring.human_approval import HumanApprovalSystem
from valetts.training.monitoring.safety import SafetyValidator

logger = logging.getLogger(__name__)


class LLMTrainingMonitor:
    """
    Monitor principal para análise de treinamento assistida por LLM.

    Coordena coleta de métricas, análise pelo Claude 4 Sonnet,
    validação de segurança e aplicação de mudanças dinâmicas.
    """

    def __init__(self, config: LLMMonitorConfig):
        """
        Inicializa o monitor LLM.

        Args:
            config: Configuração do sistema de monitoramento
        """
        self.config = config
        self.safety_validator = SafetyValidator(config)

        # 🔗 Novos sistemas integrados
        self.human_approval = HumanApprovalSystem(timeout_seconds=60)
        self.telemetry = HistoricalTelemetryCollector(max_history_size=100)

        # Configuração da API
        self.api_key = os.getenv(config.api_key_env)
        if config.enabled and not self.api_key:
            raise ValueError(f"API key não encontrada: {config.api_key_env}")

        # Histórico de análises
        self.analysis_history: List[LLMAnalysisResponse] = []

        # Configuração dinâmica
        self.dynamic_config_path = Path(config.dynamic_config_path)
        self.original_config_backup: Optional[Dict] = None

        # Estatísticas
        self.total_analyses = 0
        self.accepted_suggestions = 0
        self.rejected_suggestions = 0

        logger.info(f"🤖 LLM Monitor inicializado - Enabled: {config.enabled}")

    def should_analyze(self, epoch: int, step: Optional[int] = None) -> bool:
        """
        Verifica se deve fazer análise nesta época/step.

        Args:
            epoch: Época atual
            step: Step atual (opcional)

        Returns:
            True se deve analisar
        """
        if not self.config.enabled:
            return False

        # Análise por epochs
        if self.config.monitor_every_epochs > 0:
            return epoch % self.config.monitor_every_epochs == 0

        # Análise por steps
        if self.config.monitor_every_steps and step:
            return step % self.config.monitor_every_steps == 0

        return False

    def should_generate_report(self, epoch: int) -> bool:
        """
        Verifica se deve gerar relatório informativo nesta época.

        Args:
            epoch: Época atual

        Returns:
            True se deve gerar relatório
        """
        if not self.config.enabled or not getattr(
            self.config, "generate_reports", False
        ):
            return False

        report_frequency = getattr(self.config, "report_every_epochs", 5)
        return epoch > 0 and epoch % report_frequency == 0

    def add_epoch_metrics(self, metrics: Dict[str, float], epoch: int) -> None:
        """Adiciona métricas da época ao histórico de telemetria."""
        self.telemetry.add_epoch_metrics(metrics, epoch)

    def analyze_training_progress(
        self,
        epoch: int,
        metrics: Dict[str, float],
        current_config: Dict[str, Any],
        model_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[LLMAnalysisResponse]:
        """
        Analisa progresso do treinamento usando LLM.

        Args:
            epoch: Época atual
            metrics: Métricas de treinamento
            current_config: Configuração atual
            model_info: Informações do modelo (opcional)

        Returns:
            Resposta da análise LLM ou None se erro
        """
        if not self.config.enabled:
            return None

        try:
            # 🔗 Adicionar métricas ao histórico de telemetria
            self.add_epoch_metrics(metrics, epoch)

            # Preparar dados para análise (agora com contexto histórico)
            analysis_data = self._prepare_analysis_data(
                epoch, metrics, current_config, model_info
            )

            # Gerar prompt especializado
            prompt = self._generate_analysis_prompt(analysis_data)

            # Fazer requisição para Claude via OpenRoutes
            response = self._call_llm_api(prompt)

            if response:
                # Salvar análise no histórico
                self._save_analysis_to_history(response)
                self.total_analyses += 1

                logger.info(f"✅ Análise LLM concluída - Época {epoch}")
                return response

        except Exception as e:
            logger.error(f"❌ Erro na análise LLM: {e}")

        return None

    def generate_training_report(
        self,
        epoch: int,
        metrics: Dict[str, float],
        current_config: Dict[str, Any],
        model_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Gera relatório informativo do progresso de treinamento.

        Args:
            epoch: Época atual
            metrics: Métricas de treinamento
            current_config: Configuração atual
            model_info: Informações do modelo (opcional)

        Returns:
            Relatório em texto livre ou None se erro
        """
        if not self.config.enabled or not getattr(
            self.config, "generate_reports", False
        ):
            return None

        try:
            # Adicionar métricas ao histórico de telemetria
            self.add_epoch_metrics(metrics, epoch)

            # Preparar dados para o relatório
            analysis_data = self._prepare_analysis_data(
                epoch, metrics, current_config, model_info
            )

            # Gerar prompt para relatório informativo
            prompt = self._generate_report_prompt(analysis_data)

            # Fazer requisição para Claude via OpenRoutes
            response = self._call_llm_report_api(prompt)

            if response:
                logger.info(f"📊 Relatório informativo gerado - Época {epoch}")
                return response

        except Exception as e:
            logger.error(f"❌ Erro ao gerar relatório: {e}")

        return None

    def apply_suggestions(
        self,
        analysis: LLMAnalysisResponse,
        current_config: Dict[str, Any],
        epoch: int,
    ) -> Dict[str, Any]:
        """
        Aplica sugestões do LLM com validação de segurança.

        Args:
            analysis: Resposta da análise LLM
            current_config: Configuração atual
            epoch: Época atual

        Returns:
            Nova configuração atualizada
        """
        if not analysis.config_suggestions:
            return current_config

        # Validar sugestões
        validated_suggestions, warnings = (
            self.safety_validator.validate_suggestions(
                analysis.config_suggestions, current_config
            )
        )

        # Log warnings
        for warning in warnings:
            logger.warning(warning)

        # Verificar mudanças críticas
        is_critical, critical_changes = (
            self.safety_validator.is_critical_change_batch(
                validated_suggestions, current_config
            )
        )

        if is_critical and self.config.require_human_approval:
            logger.warning(
                f"🚨 Mudanças críticas detectadas que requerem aprovação humana:\n"
                + "\n".join(critical_changes)
            )

            # 🔗 Sistema de aprovação humana integrado
            try:
                approved, final_config = self.human_approval.request_approval(
                    epoch=epoch,
                    current_config=current_config,
                    suggested_changes=validated_suggestions,
                    llm_analysis=analysis.analysis_summary,
                )

                if approved:
                    logger.info("✅ Mudanças aprovadas pelo usuário")
                    return final_config
                else:
                    logger.info("❌ Mudanças rejeitadas pelo usuário")
                    return current_config

            except Exception as e:
                logger.error(f"❌ Erro na aprovação humana: {e}")
                # Fallback: aplicar apenas mudanças não-críticas
                validated_suggestions = {
                    k: v
                    for k, v in validated_suggestions.items()
                    if not self.config.is_critical_change(
                        k, current_config.get(k, 0), v
                    )
                }

        # Aplicar mudanças validadas
        if validated_suggestions:
            updated_config = current_config.copy()
            updated_config.update(validated_suggestions)

            # Salvar configuração dinâmica
            self._save_dynamic_config(updated_config)

            # Registrar mudanças
            self.safety_validator.record_change(epoch, validated_suggestions)
            self.accepted_suggestions += len(validated_suggestions)

            logger.info(
                f"🔄 Aplicadas {len(validated_suggestions)} sugestões LLM na época {epoch}"
            )

            for param, value in validated_suggestions.items():
                logger.info(
                    f"   {param}: {current_config.get(param)} → {value}"
                )

            return updated_config

        else:
            logger.info("ℹ️ Nenhuma sugestão válida para aplicar")
            return current_config

    def _prepare_analysis_data(
        self,
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        model_info: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Prepara dados para análise LLM com contexto histórico rico."""
        # Filtrar apenas métricas monitoradas
        filtered_metrics = {
            k: v
            for k, v in metrics.items()
            if k in self.config.metrics_to_monitor
        }

        # 🔗 Obter dados contextuais da telemetria histórica
        telemetry_data = self.telemetry.get_contextual_data(epoch)

        # Contexto de análises LLM anteriores
        llm_analysis_context = []
        if len(self.analysis_history) > 0:
            recent_analyses = self.analysis_history[
                -self.config.include_context_epochs :
            ]
            llm_analysis_context = [
                {
                    "epoch": a.epoch,
                    "status": a.overall_status,
                    "summary": a.analysis_summary,
                }
                for a in recent_analyses
            ]

        return {
            "current_epoch": epoch,
            "metrics": filtered_metrics,
            "config": config,
            "model_info": model_info or {},
            "safe_ranges": self.config.get_safe_ranges(),
            "alert_thresholds": {
                "loss_spike": self.config.loss_spike_threshold,
                "grad_norm": self.config.grad_norm_threshold,
                "memory_usage": self.config.memory_usage_threshold,
            },
            # 🔗 Contexto histórico enriquecido
            "historical_telemetry": telemetry_data,
            "llm_analysis_context": llm_analysis_context,
            "timestamp": datetime.now().isoformat(),
        }

    def _generate_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """Gera prompt especializado para análise VITS2 com contexto temporal."""

        # Determinar fase do treinamento baseado na época
        epoch = data["current_epoch"]
        if epoch < 5:
            phase = "INICIALIZAÇÃO"
        elif epoch < 20:
            phase = "ESTABILIZAÇÃO"
        elif epoch < 50:
            phase = "CONVERGÊNCIA_INICIAL"
        else:
            phase = "REFINAMENTO"

        prompt = f"""Você é um especialista em treinamento de modelos VITS2 (Variational Inference Text-to-Speech).
Analise os dados de treinamento fornecidos considerando o contexto temporal e as características específicas do VITS2.

## CONTEXTO DO TREINAMENTO

**ÉPOCA ATUAL:** {epoch} (Fase: {phase})
**Timestamp:** {data['timestamp']}

## CONHECIMENTO ESPECÍFICO VITS2:

### Comportamento Normal por Fase:
- **ÉPOCAS 0-5 (INICIALIZAÇÃO):**
  * KL Loss = Infinity é NORMAL (posterior/prior não alinhados)
  * Mel Loss alta (>5.0) é esperada
  * Discriminadores podem não estar ativos ainda

- **ÉPOCAS 5-20 (ESTABILIZAÇÃO):**
  * KL Loss deve começar a diminuir gradualmente
  * Mel Loss deve mostrar tendência de queda
  * Adversarial loss deve se tornar ativa

- **ÉPOCAS 20+ (CONVERGÊNCIA):**
  * Todas as losses devem convergir
  * KL Loss < 1.0 é bom sinal
  * Mel Loss < 2.0 é excelente

### Losses VITS2:
- **mel_loss:** Reconstrução mel-spectrogram (target: <2.0)
- **kl_loss:** KL divergence VAE (NORMAL ser infinity no início!)
- **adv_loss:** Adversarial discriminator (pode demorar para ativar)
- **fm_loss:** Feature matching (estabiliza training adversarial)
- **duration_loss:** Alinhamento texto-áudio

## DADOS ATUAIS

### Métricas da Época {epoch}:
{json.dumps(data['metrics'], indent=2)}

### Configuração Atual:
{json.dumps(data['config'], indent=2)}

### Ranges Seguros:
{json.dumps(data['safe_ranges'], indent=2)}

### Histórico (últimas épocas):
{json.dumps(data['historical_telemetry'], indent=2)}

### Análises LLM Anteriores:
{json.dumps(data['llm_analysis_context'], indent=2)}

## INSTRUÇÕES DE ANÁLISE

Considere SEMPRE a época atual ({epoch}) e a fase do treinamento ({phase}) para suas recomendações.

### Regras Específicas por Fase:
- **INICIALIZAÇÃO (0-5):** Foque em estabilidade, KL infinity é OK
- **ESTABILIZAÇÃO (5-20):** Monitor KL convergência, ajustar LR gradualmente
- **CONVERGÊNCIA (20+):** Otimizar todas as losses, refinamento fino

Responda APENAS com um JSON válido seguindo EXATAMENTE esta estrutura:

{{
    "overall_status": "healthy|warning|critical",
    "confidence_score": 0.95,
    "analysis_summary": "Resumo da análise em português",
    "config_suggestions": {{
        "learning_rate": 0.001,
        "batch_size": 32,
        "loss_weight_mel": 1.0
    }},
    "observations": [
        "Observação específica 1",
        "Observação específica 2"
    ],
    "recommended_actions": [
        "Ação recomendada 1",
        "Ação recomendada 2"
    ],
    "alerts": [
        "Alerta importante se houver"
    ],
    "next_checkpoints": [
        "Verificar na época X",
        "Monitorar métrica Y"
    ],
    "timestamp": "{data['timestamp']}",
    "epoch": {data['current_epoch']},
    "model_used": "claude-3-5-sonnet"
}}

## DIRETRIZES IMPORTANTES

1. **SEGURANÇA**: Sugira apenas mudanças dentro dos ranges seguros fornecidos
2. **GRADUALIDADE**: Evite mudanças abruptas (max 2x para LR, batch size)
3. **CONTEXTO**: Considere o histórico e tendências das métricas
4. **MÉTRICAS VITS2**: Foque em mel_loss, kl_loss, adversarial_loss
5. **ESTABILIDADE**: Priorize estabilidade sobre convergência rápida

Responda APENAS com o JSON, sem texto adicional."""

        return prompt

    def _generate_report_prompt(self, data: Dict[str, Any]) -> str:
        """Gera prompt especializado para relatórios informativos."""

        # Determinar fase do treinamento
        epoch = data["current_epoch"]
        if epoch < 5:
            phase = "INICIALIZAÇÃO"
        elif epoch < 20:
            phase = "ESTABILIZAÇÃO"
        elif epoch < 50:
            phase = "CONVERGÊNCIA_INICIAL"
        else:
            phase = "REFINAMENTO"

        prompt = f"""Você é um especialista em treinamento de modelos VITS2 (Variational Inference Text-to-Speech).
Gere um RELATÓRIO INFORMATIVO detalhado sobre o progresso do treinamento, explicando cada métrica e tendência de forma clara e educativa.

## CONTEXTO DO TREINAMENTO

**ÉPOCA ATUAL:** {epoch} (Fase: {phase})
**Timestamp:** {data['timestamp']}

## DADOS DO TREINAMENTO

### Métricas da Época {epoch}:
{json.dumps(data['metrics'], indent=2)}

### Histórico das Últimas Épocas:
{json.dumps(data['historical_telemetry'], indent=2)}

### Configuração Atual:
Learning Rate: {data['config'].get('learning_rate', 'N/A')}
Batch Size: {data['config'].get('batch_size', 'N/A')}
Mel Loss Weight: {data['config'].get('mel_loss_weight', 'N/A')}

## INSTRUÇÕES PARA O RELATÓRIO

Crie um relatório informativo e educativo que inclua:

1. **RESUMO GERAL**: Estado atual do treinamento em uma frase
2. **ANÁLISE DAS MÉTRICAS**: Explique cada perda e o que significa:
   - mel_loss: Como está a qualidade de reconstrução do mel-spectrograma
   - kl_loss: Comportamento da divergência KL (normal ser alta no início)
   - adv_loss: Status do treinamento adversarial
   - outras losses relevantes
3. **TENDÊNCIAS**: Como as métricas estão evoluindo ao longo do tempo
4. **FASE ATUAL**: O que é esperado nesta fase ({phase}) do treinamento
5. **PONTOS DE ATENÇÃO**: Aspectos que merecem observação
6. **PRÓXIMOS MARCOS**: O que esperar nas próximas épocas

## DIRETRIZES

- Use linguagem clara e educativa
- Explique o SIGNIFICADO de cada métrica, não apenas os valores
- Compare com o que é ESPERADO para esta fase
- Destaque TENDÊNCIAS positivas ou preocupantes
- Seja específico sobre VITS2 e suas características
- Use emojis para tornar mais visual: 📈 📉 ⚠️ ✅ 🎯

Responda em TEXTO LIVRE (não JSON), de forma conversacional e informativa."""

        return prompt

    def _call_llm_report_api(self, prompt: str) -> Optional[str]:
        """Faz chamada para API do Claude para gerar relatórios informativos."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://valetts.dev",
                "X-Title": "ValeTTS Training Reporter",
            }

            payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 3000,  # Mais tokens para relatórios detalhados
                "temperature": 0.3,  # Mais criativo para relatórios
            }

            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )

            response.raise_for_status()

            # Extrair resposta
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            return content

        except requests.RequestException as e:
            logger.error(f"❌ Erro na requisição API para relatório: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Erro inesperado na API LLM para relatório: {e}")
            return None

    def _call_llm_api(self, prompt: str) -> Optional[LLMAnalysisResponse]:
        """Faz chamada para API do Claude via OpenRoutes."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://valetts.dev",
                "X-Title": "ValeTTS Training Monitor",
            }

            payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.1,  # Baixa temperatura para consistência
            }

            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )

            response.raise_for_status()

            # Extrair resposta
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Parse JSON da resposta
            try:
                analysis_data = json.loads(content)
                return LLMAnalysisResponse(**analysis_data)
            except json.JSONDecodeError as e:
                logger.error(f"❌ Erro parsing JSON da resposta LLM: {e}")
                logger.debug(f"Resposta bruta: {content}")
                return None

        except requests.RequestException as e:
            logger.error(f"❌ Erro na requisição API: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Erro inesperado na API LLM: {e}")
            return None

    def _save_analysis_to_history(self, analysis: LLMAnalysisResponse) -> None:
        """Salva análise no histórico."""
        self.analysis_history.append(analysis)

        # Manter apenas histórico recente
        if len(self.analysis_history) > self.config.max_history_entries:
            self.analysis_history = self.analysis_history[
                -self.config.max_history_entries :
            ]

    def _save_dynamic_config(self, config: Dict[str, Any]) -> None:
        """Salva configuração dinâmica em arquivo."""
        try:
            # Criar diretório se não existir
            self.dynamic_config_path.parent.mkdir(parents=True, exist_ok=True)

            # Backup da configuração original (primeira vez)
            if (
                self.original_config_backup is None
                and self.dynamic_config_path.exists()
            ):
                with open(self.dynamic_config_path, "r") as f:
                    self.original_config_backup = yaml.safe_load(f)

            # Salvar nova configuração
            with open(self.dynamic_config_path, "w") as f:
                yaml.dump(config, f, indent=2, allow_unicode=True)

            logger.debug(
                f"💾 Configuração dinâmica salva: {self.dynamic_config_path}"
            )

        except Exception as e:
            logger.error(f"❌ Erro salvando configuração dinâmica: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do monitor."""
        # Calcular sugestões aplicadas/rejeitadas do histórico
        suggestions_applied = 0
        suggestions_rejected = 0
        critical_changes = 0

        for analysis in self.analysis_history:
            if hasattr(analysis, "config_suggestions"):
                suggestions_count = len(analysis.config_suggestions)
                # Por enquanto, assumir que sugestões críticas são rejeitadas
                if analysis.overall_status == "critical":
                    suggestions_rejected += suggestions_count
                    critical_changes += 1
                else:
                    suggestions_applied += suggestions_count

        return {
            "total_analyses": self.total_analyses,
            "suggestions_applied": suggestions_applied,
            "suggestions_rejected": suggestions_rejected,
            "critical_changes": critical_changes,
            "acceptance_rate": (
                suggestions_applied / max(1, self.total_analyses)
            ),
            "history_length": len(self.analysis_history),
            "enabled": self.config.enabled,
        }
