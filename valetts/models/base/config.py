"""Base configuration classes for ValeTTS models."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from omegaconf import OmegaConf


@dataclass
class BaseModelConfig:
    """Base configuration class for all ValeTTS models."""

    # Model architecture
    name: str = "base_model"
    version: str = "1.0.0"

    # Audio settings
    sample_rate: int = 22050
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: Optional[float] = None

    # Model dimensions
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1

    # Training settings
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    batch_size: int = 32
    gradient_clip_val: float = 1.0

    # Device settings
    device: str = "auto"  # auto, cpu, cuda, mps
    precision: Union[str, int] = 16  # 16, 32, "bf16"

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Additional config fields
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation."""
        if self.f_max is None:
            self.f_max = self.sample_rate // 2

        # Validate audio settings
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.win_length <= 0:
            raise ValueError("win_length must be positive")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be positive")

        # Validate model dimensions
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")

        # Auto-detect device if needed
        if self.device == "auto":
            self.device = self._auto_detect_device()

    def _auto_detect_device(self) -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BaseModelConfig":
        """Load configuration from YAML file."""
        cfg = OmegaConf.load(yaml_path)
        return cls(**cfg)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseModelConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return OmegaConf.structured(self)

    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        cfg = OmegaConf.structured(self)
        OmegaConf.save(cfg, yaml_path)

    def update(self, **kwargs) -> "BaseModelConfig":
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.extras[key] = value
        return self

    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio-related configuration parameters."""
        return {
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "n_fft": self.n_fft,
            "n_mels": self.n_mels,
            "f_min": self.f_min,
            "f_max": self.f_max,
        }

    def get_model_config(self) -> Dict[str, Any]:
        """Get model architecture configuration parameters."""
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
        }

    def get_training_config(self) -> Dict[str, Any]:
        """Get training-related configuration parameters."""
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "gradient_clip_val": self.gradient_clip_val,
        }


@dataclass
class AudioConfig:
    """Configuration for audio processing."""

    sample_rate: int = 22050
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: Optional[float] = None
    power: float = 1.0
    normalized: bool = True
    preemphasis: float = 0.97

    def __post_init__(self):
        """Post-initialization validation."""
        if self.f_max is None:
            self.f_max = self.sample_rate // 2


@dataclass
class TextConfig:
    """Configuration for text processing."""

    vocab_size: int = 256
    pad_token_id: int = 0
    unk_token_id: int = 1
    eos_token_id: int = 2
    max_length: int = 512
    use_phonemes: bool = True
    language: str = "pt-br"
    supported_languages: List[str] = field(default_factory=lambda: ["pt-br", "en-us"])

    def __post_init__(self):
        """Post-initialization validation."""
        if self.language not in self.supported_languages:
            raise ValueError(f"Language {self.language} not in supported languages: {self.supported_languages}")
