"""VITS2 model configuration."""

from dataclasses import dataclass, field
from typing import List, Optional

from valetts.models.base.config import BaseModelConfig


@dataclass
class VITS2Config(BaseModelConfig):
    """Configuration for VITS2 model."""

    # Override base model info
    name: str = "vits2"
    version: str = "2.0.0"

    # Audio parameters (aliases from base config)
    mel_channels: int = 80  # n_mels from base config
    hop_length: int = 256  # From base config

    # Text encoder configuration
    text_vocab_size: int = 256
    text_hidden_dim: int = 192
    text_encoder_hidden_dim: int = 192  # Alias for text_hidden_dim
    text_encoder_layers: int = 6
    text_encoder_attention_heads: int = 2
    text_encoder_filter_size: int = 768
    text_filter_channels: int = 768
    text_num_heads: int = 2
    text_num_layers: int = 6
    text_kernel_size: int = 3
    text_dropout: float = 0.1
    max_text_length: int = 1000

    # Posterior encoder configuration
    posterior_in_channels: int = 513  # n_fft // 2 + 1
    posterior_out_channels: int = 192
    posterior_hidden_channels: int = 192
    posterior_encoder_hidden_dim: int = 192  # Alias
    posterior_encoder_layers: int = 16
    posterior_kernel_size: int = 5
    posterior_dilation_rate: int = 1
    posterior_num_layers: int = 16

    # Flow configuration
    flow_flows: int = 4
    flow_kernel_size: int = 5
    flow_base_dilation: int = 1
    flow_num_layers: int = 4
    flow_gin_channels: int = 256
    flow_hidden_channels: int = 192
    flow_dilation_rate: int = 1
    n_flow_blocks: int = 4
    n_flow_layers: int = 4
    latent_dim: int = 192

    # Generator configuration
    generator_initial_channels: int = 512
    decoder_hidden_dim: int = 512  # Alias for generator
    generator_resblock_type: str = "1"  # "1" or "2"
    generator_resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    generator_resblock_kernel_sizes: List[int] = field(
        default_factory=lambda: [3, 7, 11]
    )
    upsample_rates: List[int] = field(
        default_factory=lambda: [8, 8, 2, 2]
    )
    upsample_kernel_sizes: List[int] = field(
        default_factory=lambda: [16, 16, 4, 4]
    )
    generator_upsample_rates: List[int] = field(
        default_factory=lambda: [8, 8, 2, 2]
    )
    generator_upsample_initial_channel: int = 512
    generator_upsample_kernel_sizes: List[int] = field(
        default_factory=lambda: [16, 16, 4, 4]
    )
    resblock_layers: int = 3
    resblock_kernel_sizes: List[int] = field(
        default_factory=lambda: [3, 7, 11]
    )
    resblock_dilations: List[int] = field(
        default_factory=lambda: [1, 3, 5]
    )

    # Discriminator configuration
    # Multi-Scale Discriminator
    msd_use_spectral_norm: bool = False
    msd_scale_downsample_pooling: str = "AvgPool1d"
    msd_scale_downsample_pooling_params: dict = field(
        default_factory=lambda: {"kernel_size": 4, "stride": 2, "padding": 2}
    )
    msd_scale_discriminator_params: dict = field(
        default_factory=lambda: {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        }
    )
    msd_scales: int = 3

    # Multi-Period Discriminator
    mpd_periods: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    mpd_use_spectral_norm: bool = False
    mpd_discriminator_params: dict = field(
        default_factory=lambda: {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        }
    )

    # Duration predictor configuration
    duration_predictor_kernel_size: int = 3
    duration_predictor_filter_size: int = 256  # Alias
    duration_predictor_filter_channels: int = 256
    duration_predictor_dropout: float = 0.5

    # Stochastic duration predictor configuration
    stochastic_duration_predictor_kernel_size: int = 3
    stochastic_duration_predictor_dropout: float = 0.5
    stochastic_duration_predictor_flows: int = 4
    stochastic_duration_predictor_dds: int = 4

    # Loss configuration
    lambda_adv: float = 1.0
    lambda_mel: float = 45.0
    lambda_feat: float = 2.0
    lambda_dur: float = 1.0
    lambda_kl: float = 1.0
    # Aliases for loss weights
    adversarial_loss_weight: float = 1.0
    feature_matching_loss_weight: float = 2.0
    mel_loss_weight: float = 45.0
    duration_loss_weight: float = 1.0
    kl_loss_weight: float = 1.0

    # Speaker and prosody configuration
    n_speakers: int = 1  # Single speaker by default
    speaker_embedding_dim: int = 256
    use_prosody_encoder: bool = False
    prosody_dim: int = 128

    # Training configuration
    segment_size: int = 8192
    use_mel_loss: bool = True
    use_kl_loss: bool = True
    use_duration_loss: bool = True
    use_feature_matching_loss: bool = True
    inference_only: bool = False  # Include discriminators by default

    # Inference configuration
    inference_noise_scale: float = 0.667
    inference_noise_scale_w: float = 0.8
    inference_length_scale: float = 1.0

    def __post_init__(self):
        """Post-initialization validation for VITS2."""
        super().__post_init__()

        # Validate generator configuration
        if len(self.generator_upsample_rates) != len(self.generator_upsample_kernel_sizes):
            raise ValueError("Number of upsample rates and kernel sizes must match")

        # Validate resblock configuration
        if len(self.generator_resblock_dilation_sizes) != len(self.generator_resblock_kernel_sizes):
            raise ValueError("Number of resblock dilation sizes and kernel sizes must match")

        # Validate discriminator periods
        if len(set(self.mpd_periods)) != len(self.mpd_periods):
            raise ValueError("MPD periods must be unique")

        # Validate loss weights
        loss_weights = [
            self.lambda_adv, self.lambda_mel, self.lambda_feat,
            self.lambda_dur, self.lambda_kl
        ]
        if any(w < 0 for w in loss_weights):
            raise ValueError("All loss weights must be non-negative")

        # Validate inference scales
        if not 0.0 <= self.inference_noise_scale <= 2.0:
            raise ValueError("inference_noise_scale must be between 0.0 and 2.0")
        if not 0.0 <= self.inference_noise_scale_w <= 2.0:
            raise ValueError("inference_noise_scale_w must be between 0.0 and 2.0")
        if not 0.1 <= self.inference_length_scale <= 3.0:
            raise ValueError("inference_length_scale must be between 0.1 and 3.0")

    def get_generator_config(self) -> dict:
        """Get generator-specific configuration."""
        return {
            "initial_channels": self.generator_initial_channels,
            "resblock_type": self.generator_resblock_type,
            "resblock_dilation_sizes": self.generator_resblock_dilation_sizes,
            "resblock_kernel_sizes": self.generator_resblock_kernel_sizes,
            "upsample_rates": self.generator_upsample_rates,
            "upsample_initial_channel": self.generator_upsample_initial_channel,
            "upsample_kernel_sizes": self.generator_upsample_kernel_sizes,
        }

    def get_discriminator_config(self) -> dict:
        """Get discriminator-specific configuration."""
        return {
            "msd": {
                "use_spectral_norm": self.msd_use_spectral_norm,
                "scale_downsample_pooling": self.msd_scale_downsample_pooling,
                "scale_downsample_pooling_params": self.msd_scale_downsample_pooling_params,
                "scale_discriminator_params": self.msd_scale_discriminator_params,
                "scales": self.msd_scales,
            },
            "mpd": {
                "periods": self.mpd_periods,
                "use_spectral_norm": self.mpd_use_spectral_norm,
                "discriminator_params": self.mpd_discriminator_params,
            }
        }

    def get_loss_config(self) -> dict:
        """Get loss-specific configuration."""
        return {
            "lambda_adv": self.lambda_adv,
            "lambda_mel": self.lambda_mel,
            "lambda_feat": self.lambda_feat,
            "lambda_dur": self.lambda_dur,
            "lambda_kl": self.lambda_kl,
            "use_mel_loss": self.use_mel_loss,
            "use_kl_loss": self.use_kl_loss,
            "use_duration_loss": self.use_duration_loss,
            "use_feature_matching_loss": self.use_feature_matching_loss,
        }

    def get_inference_config(self) -> dict:
        """Get inference-specific configuration."""
        config = super().get_inference_config()
        config.update({
            "noise_scale": self.inference_noise_scale,
            "noise_scale_w": self.inference_noise_scale_w,
            "length_scale": self.inference_length_scale,
            "segment_size": self.segment_size,
        })
        return config
