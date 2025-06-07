#!/usr/bin/env python3
"""
Script de treinamento para ValeTTS.

Uso:
    # Treinamento básico
    python scripts/train.py --config configs/training/vits2_training.yaml

    # Com override de parâmetros
    python scripts/train.py --config configs/training/vits2_training.yaml \
        trainer.max_epochs=100 data.batch_size=16

    # Treinamento distribuído
    python scripts/train.py --config configs/training/vits2_training.yaml \
        trainer.accelerator=gpu trainer.devices=2
"""

import argparse
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from valetts.training.trainers import VITS2Trainer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ValeTTS Training Script")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name for logging",
    )

    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides (e.g., trainer.max_epochs=100)",
    )

    return parser.parse_args()


@hydra.main(version_base=None, config_path=None)
def train(cfg: DictConfig) -> None:
    """Training function with Hydra configuration."""
    print("Starting ValeTTS training...")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seeds for reproducibility
    if "seed" in cfg:
        pl.seed_everything(cfg.seed)

    # Initialize trainer
    trainer = VITS2Trainer(cfg)

    # Start training
    trainer.fit()

    # Save final model
    trainer.save_model()

    print("Training completed!")


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Set up Hydra with config file
    with hydra.initialize_config_dir(
        config_dir=str(config_path.parent.absolute()), version_base=None
    ):
        # Override config file name
        overrides = [f"config={config_path.stem}"]

        # Add command line overrides
        if args.overrides:
            overrides.extend(args.overrides)

        # Add experiment name if provided
        if args.experiment_name:
            overrides.append(f"experiment_name={args.experiment_name}")

        # Add checkpoint path if provided
        if args.resume_from_checkpoint:
            overrides.append(
                f"trainer.resume_from_checkpoint={args.resume_from_checkpoint}"
            )

        # Compose configuration and run training
        cfg = hydra.compose(config_name=config_path.stem, overrides=overrides)
        train(cfg)


if __name__ == "__main__":
    main()
