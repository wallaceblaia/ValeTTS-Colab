"""Base model class for all ValeTTS models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from .config import BaseModelConfig


class BaseModel(L.LightningModule, ABC):
    """Base model class for all ValeTTS models.

    This class provides common functionality for all models including:
    - Configuration management
    - Optimizer and scheduler setup
    - Logging utilities
    - Device management
    - Model saving/loading
    """

    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.to_dict())

        # Automatically enable optimizations
        self.automatic_optimization = True

        # Store validation metrics for early stopping
        self.validation_metrics = {}

    @abstractmethod
    def forward(self, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass of the model.

        This method must be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Training batch
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        pass

    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step.

        Args:
            batch: Validation batch
            batch_idx: Batch index

        Returns:
            Dictionary of validation metrics
        """
        pass

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step - defaults to validation step.

        Args:
            batch: Test batch
            batch_idx: Batch index

        Returns:
            Dictionary of test metrics
        """
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        # Default optimizer
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        # Default scheduler - cosine annealing with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-7,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        # Log learning rate
        if self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log("learning_rate", current_lr, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Clear cache to prevent memory issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_device(self) -> torch.device:
        """Get current device of the model."""
        return next(self.parameters()).device

    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters.

        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": total_params - trainable_params,
        }

    def log_parameter_counts(self) -> None:
        """Log parameter counts."""
        param_counts = self.count_parameters()
        total_mb = param_counts["total"] * 4 / (1024 * 1024)  # Assuming float32

        self.log("params/total", param_counts["total"])
        self.log("params/trainable", param_counts["trainable"])
        self.log("params/total_mb", total_mb)

    def freeze_layers(self, layer_names: Union[str, list]) -> None:
        """Freeze specific layers.

        Args:
            layer_names: Name or list of layer names to freeze
        """
        if isinstance(layer_names, str):
            layer_names = [layer_names]

        for name, param in self.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = False
                    break

    def unfreeze_layers(self, layer_names: Union[str, list]) -> None:
        """Unfreeze specific layers.

        Args:
            layer_names: Name or list of layer names to unfreeze
        """
        if isinstance(layer_names, str):
            layer_names = [layer_names]

        for name, param in self.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = True
                    break

    def get_model_summary(self) -> str:
        """Get a summary of the model architecture.

        Returns:
            String representation of model summary
        """
        param_counts = self.count_parameters()

        summary = f"""
{self.__class__.__name__} Summary:
{'=' * 50}
Configuration: {self.config.name} v{self.config.version}
Device: {self.get_device()}
Parameters:
  - Total: {param_counts['total']:,}
  - Trainable: {param_counts['trainable']:,}
  - Non-trainable: {param_counts['non_trainable']:,}
  - Size (MB): {param_counts['total'] * 4 / (1024 * 1024):.2f}
Audio Config:
  - Sample Rate: {self.config.sample_rate}
  - Hop Length: {self.config.hop_length}
  - N-FFT: {self.config.n_fft}
  - N-Mels: {self.config.n_mels}
Model Config:
  - Hidden Dim: {self.config.hidden_dim}
  - Num Layers: {self.config.num_layers}
  - Num Heads: {self.config.num_heads}
  - Dropout: {self.config.dropout}
{'=' * 50}
        """
        return summary.strip()

    def save_config(self, path: str) -> None:
        """Save model configuration to file.

        Args:
            path: Path to save configuration
        """
        self.config.save_yaml(path)

    @classmethod
    def load_from_config(cls, config_path: str, checkpoint_path: Optional[str] = None):
        """Load model from configuration file.

        Args:
            config_path: Path to configuration file
            checkpoint_path: Optional path to model checkpoint

        Returns:
            Loaded model instance
        """
        config = BaseModelConfig.from_yaml(config_path)
        model = cls(config)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])

        return model

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB.

        Returns:
            Dictionary with memory usage information
        """
        device = self.get_device()

        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
            cached = torch.cuda.memory_reserved(device) / (1024 * 1024)

            return {
                "allocated_mb": allocated,
                "cached_mb": cached,
                "total_mb": allocated + cached,
            }
        else:
            # For CPU, we can't easily get memory usage per model
            return {
                "allocated_mb": 0.0,
                "cached_mb": 0.0,
                "total_mb": 0.0,
            }

    def log_memory_usage(self) -> None:
        """Log current memory usage."""
        memory_usage = self.get_memory_usage()

        for key, value in memory_usage.items():
            self.log(f"memory/{key}", value)

    def setup_for_inference(self) -> None:
        """Setup model for inference mode."""
        self.eval()
        self.freeze()

        # Enable inference optimizations
        if hasattr(torch, "jit"):
            torch.jit.optimized_execution(True)

    def get_inference_config(self) -> Dict[str, Any]:
        """Get configuration optimized for inference.

        Returns:
            Dictionary with inference configuration
        """
        return {
            "model_name": self.config.name,
            "sample_rate": self.config.sample_rate,
            "hop_length": self.config.hop_length,
            "device": str(self.get_device()),
            "precision": self.config.precision,
        }
