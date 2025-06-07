"""
Trainer base para modelos ValeTTS.

Fornece infraestrutura comum para treinamento com suporte a:
- Monitoramento LLM opcional
- Mixed precision training
- Multi-GPU distributed training
- Gradient accumulation e clipping
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as L
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

from valetts.training.audio_sampling import AudioSampleGenerator
from valetts.training.monitoring import LLMMonitorConfig, LLMTrainingMonitor

logger = logging.getLogger(__name__)


class BaseTrainer(L.LightningModule, ABC):
    """
    Trainer base para modelos ValeTTS.

    Fornece funcionalidade comum de treinamento com integração
    opcional do sistema de monitoramento LLM.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        llm_monitor_config: Optional[LLMMonitorConfig] = None,
    ):
        """
        Inicializa o trainer base.

        Args:
            model: Modelo a ser treinado
            config: Configuração de treinamento
            llm_monitor_config: Configuração do monitoramento LLM (opcional)
        """
        super().__init__()

        self.model = model
        self.config = config
        self.save_hyperparameters(config)

        # Sistema de Monitoramento LLM (opcional)
        self.llm_monitor = None
        if llm_monitor_config and llm_monitor_config.enabled:
            self.llm_monitor = LLMTrainingMonitor(llm_monitor_config)
            logger.info("🤖 Monitoramento LLM ativado!")

        # 🎵 Sistema de Geração de Amostras (opcional)
        self.audio_sampler = None
        if config.get("generate_samples", True):
            self.audio_sampler = AudioSampleGenerator(
                output_dir="samples",
                sample_rate=config.get("sample_rate", 22050),
            )
            logger.info("🎵 Gerador de amostras ativado!")

        # Configurações de treinamento
        self.learning_rate = config.get("learning_rate", 2e-4)
        self.batch_size = config.get("batch_size", 16)
        self.accumulate_grad_batches = config.get("accumulate_grad_batches", 1)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)

        # Métricas de monitoramento
        self.epoch_metrics: Dict[str, float] = {}
        self.step_metrics: Dict[str, float] = {}

        # Cronômetros
        self.epoch_start_time = None

    def on_train_epoch_start(self) -> None:
        """Callback no início de cada época."""
        self.epoch_start_time = time.time()
        self.epoch_metrics = {}

        # Monitoramento LLM: verificar se deve analisar
        if self.llm_monitor and self.llm_monitor.should_analyze(
            self.current_epoch
        ):
            logger.info(
                f"🤖 Época {self.current_epoch}: Análise LLM programada"
            )

    def on_train_epoch_end(self) -> None:
        """Callback no fim de cada época."""
        # Calcular tempo da época
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_metrics["epoch_time"] = epoch_time

        # Adicionar outras métricas
        self.epoch_metrics["learning_rate"] = self.get_current_lr()
        self.epoch_metrics["memory_usage"] = self.get_memory_usage()

        # Monitoramento LLM: executar análise se programada
        if self.llm_monitor and self.llm_monitor.should_analyze(
            self.current_epoch
        ):
            self._run_llm_analysis()

        # Relatórios informativos LLM: gerar se programado
        if self.llm_monitor and self.llm_monitor.should_generate_report(
            self.current_epoch
        ):
            self._generate_training_report()

        # 🎵 Geração de amostras de áudio (se configurado)
        if self.audio_sampler and self._should_generate_samples():
            self._generate_audio_samples()

        # Log métricas
        self._log_epoch_metrics()

    def _run_llm_analysis(self) -> None:
        """Executa análise LLM e aplica sugestões."""
        try:
            logger.info(
                f"🤖 Executando análise LLM - Época {self.current_epoch}"
            )

            # Coletar configuração atual
            current_config = {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "max_grad_norm": self.max_grad_norm,
                **self._get_model_specific_config(),
            }

            # Executar análise
            analysis = self.llm_monitor.analyze_training_progress(
                epoch=self.current_epoch,
                metrics=self.epoch_metrics,
                current_config=current_config,
                model_info=self._get_model_info(),
            )

            if analysis:
                logger.info(
                    f"✅ Análise concluída - Status: {analysis.overall_status}"
                )

                # Aplicar sugestões
                updated_config = self.llm_monitor.apply_suggestions(
                    analysis, current_config, self.current_epoch
                )

                # Aplicar mudanças ao trainer
                self._apply_config_changes(updated_config, current_config)

                # Log análise
                self._log_llm_analysis(analysis)

            else:
                logger.warning("❌ Erro na análise LLM")

        except Exception as e:
            logger.error(f"❌ Erro no monitoramento LLM: {e}")

    def _generate_training_report(self) -> None:
        """Gera relatório informativo do progresso de treinamento."""
        try:
            logger.info(
                f"📊 Gerando relatório informativo - Época {self.current_epoch}"
            )

            # Coletar configuração atual
            current_config = {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "max_grad_norm": self.max_grad_norm,
                **self._get_model_specific_config(),
            }

            # Gerar relatório
            report = self.llm_monitor.generate_training_report(
                epoch=self.current_epoch,
                metrics=self.epoch_metrics,
                current_config=current_config,
                model_info=self._get_model_info(),
            )

            if report:
                # Salvar relatório em arquivo
                self._save_training_report(report)

                # Log do relatório na tela
                logger.info("=" * 80)
                logger.info(
                    f"📋 RELATÓRIO DE TREINAMENTO - ÉPOCA {self.current_epoch}"
                )
                logger.info("=" * 80)
                logger.info(report)
                logger.info("=" * 80)
            else:
                logger.warning("❌ Erro ao gerar relatório informativo")

        except Exception as e:
            logger.error(f"❌ Erro ao gerar relatório: {e}")

    def _save_training_report(self, report: str) -> None:
        """Salva relatório de treinamento em arquivo."""
        try:
            import json
            from datetime import datetime

            # Criar diretório de relatórios se não existir
            reports_dir = Path("logs/reports")
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Nome do arquivo com timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_report_epoch_{self.current_epoch:03d}_{timestamp}.md"
            report_path = reports_dir / filename

            # Criar conteúdo do arquivo com metadados
            content = f"""# Relatório de Treinamento VITS2 - Época {self.current_epoch}

**Data:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Modelo:** VITS2
**Dataset:** Dataset-Unificado
**Época:** {self.current_epoch}

---

{report}

---

**Métricas da Época:**
```json
{json.dumps(self.epoch_metrics, indent=2)}
```

**Arquivo gerado automaticamente pelo sistema de monitoramento LLM ValeTTS**
"""

            # Salvar arquivo
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"💾 Relatório salvo: {report_path}")

        except Exception as e:
            logger.error(f"❌ Erro ao salvar relatório: {e}")

    def _apply_config_changes(
        self, updated_config: Dict[str, Any], current_config: Dict[str, Any]
    ) -> None:
        """
        Aplica mudanças de configuração sugeridas pelo LLM.

        Args:
            updated_config: Nova configuração
            current_config: Configuração atual
        """
        changes_applied = []

        # Learning Rate
        if "learning_rate" in updated_config:
            new_lr = updated_config["learning_rate"]
            if new_lr != current_config["learning_rate"]:
                self.learning_rate = new_lr
                # Atualizar optimizer se já existir
                try:
                    optimizers = self.optimizers()
                    if isinstance(optimizers, list):
                        for optimizer in optimizers:
                            for param_group in optimizer.param_groups:
                                param_group["lr"] = new_lr
                    else:
                        for param_group in optimizers.param_groups:
                            param_group["lr"] = new_lr
                except Exception as e:
                    logger.warning(f"Não foi possível atualizar LR: {e}")
                changes_applied.append(
                    f"learning_rate: {current_config['learning_rate']:.2e} → {new_lr:.2e}"
                )

        # Batch Size (requer restart do DataLoader - loggar apenas)
        if "batch_size" in updated_config:
            new_bs = updated_config["batch_size"]
            if new_bs != current_config["batch_size"]:
                self.batch_size = new_bs
                changes_applied.append(
                    f"batch_size: {current_config['batch_size']} → {new_bs} (aplicar no próximo restart)"
                )

        # Max Grad Norm
        if "max_grad_norm" in updated_config:
            new_grad_norm = updated_config["max_grad_norm"]
            if new_grad_norm != current_config["max_grad_norm"]:
                self.max_grad_norm = new_grad_norm
                changes_applied.append(
                    f"max_grad_norm: {current_config['max_grad_norm']:.1f} → {new_grad_norm:.1f}"
                )

        # Aplicar mudanças específicas do modelo
        model_changes = self._apply_model_specific_changes(
            updated_config, current_config
        )
        changes_applied.extend(model_changes)

        # Log mudanças aplicadas
        if changes_applied:
            logger.info("🔄 Mudanças aplicadas pelo LLM:")
            for change in changes_applied:
                logger.info(f"   • {change}")
        else:
            logger.info("ℹ️ Nenhuma mudança aplicada pelo LLM")

    def get_current_lr(self) -> float:
        """Retorna learning rate atual."""
        try:
            optimizers = self.optimizers()
            if isinstance(optimizers, list):
                optimizer = optimizers[0]
            else:
                optimizer = optimizers
            return optimizer.param_groups[0]["lr"]
        except:
            return self.learning_rate

    def get_memory_usage(self) -> float:
        """Retorna uso de memória GPU como fração (0-1)."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            return max(allocated, reserved) / total
        return 0.0

    def _log_epoch_metrics(self) -> None:
        """Log métricas da época."""
        for key, value in self.epoch_metrics.items():
            self.log(f"epoch/{key}", value, on_epoch=True, prog_bar=False)

    def _log_llm_analysis(self, analysis) -> None:
        """Log análise LLM."""
        self.log(
            "llm/confidence_score", analysis.confidence_score, on_epoch=True
        )
        self.log(
            "llm/status",
            1.0 if analysis.overall_status == "healthy" else 0.0,
            on_epoch=True,
        )

        # Log observações como texto (se logger suportar)
        if hasattr(self.logger, "experiment"):
            try:
                self.logger.experiment.add_text(
                    "llm/analysis_summary",
                    analysis.analysis_summary,
                    self.current_epoch,
                )
            except:
                pass  # Alguns loggers não suportam texto

    @abstractmethod
    def _get_model_specific_config(self) -> Dict[str, Any]:
        """Retorna configuração específica do modelo para o LLM."""
        pass

    @abstractmethod
    def _apply_model_specific_changes(
        self, updated_config: Dict[str, Any], current_config: Dict[str, Any]
    ) -> List[str]:
        """Aplica mudanças específicas do modelo."""
        pass

    @abstractmethod
    def _get_model_info(self) -> Dict[str, Any]:
        """Retorna informações do modelo para análise LLM."""
        pass

    def _should_generate_samples(self) -> bool:
        """Verifica se deve gerar amostras nesta época."""
        sample_frequency = self.config.get("sample_every_n_epochs", 5)
        return self.current_epoch % sample_frequency == 0

    def _generate_audio_samples(self) -> None:
        """Gera amostras de áudio para validação qualitativa."""
        try:
            logger.info(
                f"🎵 Iniciando geração de amostras - Época {self.current_epoch}"
            )

            # Obter preprocessador de texto (método abstrato)
            text_preprocessor = self._get_text_preprocessor()

            # Gerar amostras
            summary = self.audio_sampler.generate_samples(
                model=self.model,
                epoch=self.current_epoch,
                text_preprocessor=text_preprocessor,
                device=self.device,
                num_samples_per_speaker=1,
            )

            # Log resultados
            if summary.get("total_samples", 0) > 0:
                logger.info(f"✅ Geradas {summary['total_samples']} amostras")

                # Log no TensorBoard/Wandb se disponível
                self.log(
                    "audio_samples/generated",
                    summary["total_samples"],
                    on_epoch=True,
                )
            else:
                logger.warning("⚠️ Nenhuma amostra gerada")

        except Exception as e:
            logger.error(f"❌ Erro na geração de amostras: {e}")

    @abstractmethod
    def _get_text_preprocessor(self):
        """Retorna o preprocessador de texto do modelo."""
        pass

    def configure_optimizers(self):
        """Configurar otimizadores e schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "epoch/val_loss_total",
                "frequency": 1,
            },
        }

    def on_before_optimizer_step(self, optimizer) -> None:
        """Gradient clipping antes do step do optimizer."""
        if self.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.max_grad_norm
            )

            # Salvar norma do gradiente para monitoramento
            self.step_metrics["grad_norm"] = float(grad_norm)
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=True)

    def get_llm_statistics(self) -> Optional[Dict[str, Any]]:
        """Retorna estatísticas do monitoramento LLM."""
        if self.llm_monitor:
            return self.llm_monitor.get_statistics()
        return None
