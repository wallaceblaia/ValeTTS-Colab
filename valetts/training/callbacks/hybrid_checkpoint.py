"""
Callback para sistema híbrido de checkpoints.

Implementa dois tipos de checkpoints:
1. Checkpoints recentes: Apenas os 2 últimos (se sobrescrevem)
2. Checkpoints backup: A cada 10 épocas (permanentes)
"""

import os
import shutil
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info


class HybridCheckpointCallback(pl.Callback):
    """
    Callback para gerenciar sistema híbrido de checkpoints.
    
    Mantém:
    - 2 checkpoints mais recentes (sobrescreve antigos)
    - Backups a cada N épocas (permanentes)
    """

    def __init__(
        self,
        backup_dirpath: str,
        backup_filename: str,
        backup_every_n_epochs: int = 10,
        monitor: str = "val_loss",
        mode: str = "min",
        verbose: bool = True,
    ):
        """
        Inicializa callback de checkpoint híbrido.
        
        Args:
            backup_dirpath: Diretório para checkpoints de backup
            backup_filename: Template do nome dos arquivos de backup
            backup_every_n_epochs: Frequência de backup (épocas)
            monitor: Métrica a monitorar
            mode: 'min' ou 'max' para a métrica
            verbose: Se deve imprimir logs
        """
        super().__init__()
        
        self.backup_dirpath = Path(backup_dirpath)
        self.backup_filename = backup_filename
        self.backup_every_n_epochs = backup_every_n_epochs
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        
        # Criar diretório de backup se necessário
        self.backup_dirpath.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            rank_zero_info(f"📦 HybridCheckpoint configurado:")
            rank_zero_info(f"   🗄️ Backup dir: {self.backup_dirpath}")
            rank_zero_info(f"   🔄 Backup a cada: {self.backup_every_n_epochs} épocas")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Chamado ao final de cada época de treinamento."""
        current_epoch = trainer.current_epoch
        
        # Verificar se é época de backup
        if (current_epoch + 1) % self.backup_every_n_epochs == 0:
            self._save_backup_checkpoint(trainer, pl_module, current_epoch)

    def _save_backup_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        epoch: int
    ) -> None:
        """Salva checkpoint de backup."""
        
        # Obter valor da métrica monitorada
        monitor_value = self._get_monitor_value(trainer)
        
        # Formatar nome do arquivo
        if monitor_value is not None:
            filename = self.backup_filename.format(
                epoch=epoch + 1,  # +1 porque epoch começa em 0
                **{f"epoch/{self.monitor}": monitor_value}
            )
        else:
            filename = self.backup_filename.format(
                epoch=epoch + 1,
                **{f"epoch/{self.monitor}": 0.0}
            )
        
        backup_path = self.backup_dirpath / filename
        
        try:
            # Salvar checkpoint de backup
            trainer.save_checkpoint(str(backup_path))
            
            if self.verbose:
                size_mb = backup_path.stat().st_size / 1024 / 1024
                rank_zero_info(f"🔒 Backup checkpoint salvo: {filename} ({size_mb:.1f}MB)")
                
        except Exception as e:
            rank_zero_info(f"❌ Erro ao salvar backup checkpoint: {e}")

    def _get_monitor_value(self, trainer: pl.Trainer) -> Optional[float]:
        """Obtém valor atual da métrica monitorada."""
        try:
            # Tentar obter do callback_metrics
            if self.monitor in trainer.callback_metrics:
                return float(trainer.callback_metrics[self.monitor])
            
            # Tentar obter do logged_metrics
            if hasattr(trainer, 'logged_metrics') and self.monitor in trainer.logged_metrics:
                return float(trainer.logged_metrics[self.monitor])
            
            # Buscar por versões com prefixo epoch/
            epoch_monitor = f"epoch/{self.monitor}"
            if epoch_monitor in trainer.callback_metrics:
                return float(trainer.callback_metrics[epoch_monitor])
            
            return None
            
        except Exception:
            return None

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Chamado ao final do treinamento."""
        if self.verbose:
            # Contar backups salvos
            backup_files = list(self.backup_dirpath.glob("*.ckpt"))
            total_size = sum(f.stat().st_size for f in backup_files) / 1024 / 1024
            
            rank_zero_info(f"📊 Resumo dos checkpoints de backup:")
            rank_zero_info(f"   🔒 Total de backups: {len(backup_files)}")
            rank_zero_info(f"   💾 Tamanho total: {total_size:.1f}MB")
            
            if backup_files:
                rank_zero_info(f"   📁 Arquivos:")
                for backup_file in sorted(backup_files):
                    size_mb = backup_file.stat().st_size / 1024 / 1024
                    rank_zero_info(f"      📦 {backup_file.name} ({size_mb:.1f}MB)")


def create_hybrid_checkpoint_callbacks(config: dict) -> list:
    """
    Cria callbacks para sistema híbrido de checkpoints.
    
    Args:
        config: Configuração com seções 'checkpoint' e 'checkpoint_backup'
    
    Returns:
        Lista de callbacks configurados
    """
    callbacks = []
    
    # Callback principal (2 mais recentes)
    main_checkpoint = ModelCheckpoint(
        dirpath=config['checkpoint']['dirpath'],
        filename=config['checkpoint']['filename'],
        monitor=config['checkpoint']['monitor'],
        mode=config['checkpoint']['mode'],
        save_top_k=config['checkpoint']['save_top_k'],
        save_last=config['checkpoint']['save_last'],
        every_n_epochs=config['checkpoint']['every_n_epochs'],
        auto_insert_metric_name=config['checkpoint'].get('auto_insert_metric_name', False),
        save_on_train_epoch_end=config['checkpoint'].get('save_on_train_epoch_end', True),
    )
    callbacks.append(main_checkpoint)
    
    # Callback de backup (se habilitado)
    if config.get('checkpoint_backup', {}).get('enabled', False):
        backup_config = config['checkpoint_backup']
        
        backup_checkpoint = HybridCheckpointCallback(
            backup_dirpath=backup_config['dirpath'],
            backup_filename=backup_config['filename'],
            backup_every_n_epochs=backup_config['every_n_epochs'],
            monitor=backup_config['monitor'],
            mode=backup_config['mode'],
            verbose=True,
        )
        callbacks.append(backup_checkpoint)
    
    return callbacks