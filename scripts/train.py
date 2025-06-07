#!/usr/bin/env python3
"""
Script principal de treinamento para ValeTTS.

Uso:
    python scripts/train.py --config configs/training/vits2_training.yaml
    
    # Com overrides
    python scripts/train.py --config configs/training/vits2_training.yaml \
        trainer.max_epochs=500 data.batch_size=16
        
    # Distributed training
    python scripts/train.py --config configs/training/vits2_training.yaml \
        distributed.devices=4 distributed.strategy=ddp
"""

import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from valetts.training.trainers import VITS2Trainer
from valetts.data.loaders import TTSDataLoader


@hydra.main(version_base=None, config_path="../configs/training", config_name="vits2_training")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    
    # Set random seeds for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Initialize data loader
    print("Initializing data loader...")
    dataloader = TTSDataLoader(cfg.data)
    
    # Initialize model trainer
    print("Initializing model trainer...")
    trainer_module = VITS2Trainer(cfg)
    
    # Initialize PyTorch Lightning trainer
    print("Setting up PyTorch Lightning trainer...")
    
    # Setup callbacks
    callbacks = []
    if "callbacks" in cfg:
        for callback_cfg in cfg.callbacks:
            callback = hydra.utils.instantiate(callback_cfg)
            callbacks.append(callback)
    
    # Setup logger
    logger = None
    if "logger" in cfg:
        logger = hydra.utils.instantiate(cfg.logger)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        precision=cfg.trainer.precision,
        callbacks=callbacks,
        logger=logger,
        strategy=cfg.distributed.strategy if "distributed" in cfg else "auto",
        devices=cfg.distributed.devices if "distributed" in cfg else "auto",
        num_nodes=cfg.distributed.num_nodes if "distributed" in cfg else 1,
    )
    
    # Setup data loaders
    train_loader = dataloader.train_dataloader()
    val_loader = dataloader.val_dataloader()
    
    # Start training
    print("Starting training...")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    trainer.fit(
        model=trainer_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    
    print("Training completed!")
    
    # Save final model
    if trainer.is_global_zero:
        final_model_path = Path("models") / "final_model.ckpt"
        final_model_path.parent.mkdir(exist_ok=True)
        trainer.save_checkpoint(final_model_path)
        print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main() 