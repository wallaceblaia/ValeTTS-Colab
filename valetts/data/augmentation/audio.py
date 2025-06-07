"""
Augmentação de áudio para ValeTTS.

Técnicas de augmentação para melhorar robustez do modelo:
- Pitch shifting
- Speed/tempo changes
- Noise addition
- Volume changes
- SpecAugment
"""

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import random
from typing import Optional, Tuple, Union, List
import math


class AudioAugmentation:
    """
    Sistema de augmentação de áudio para TTS.

    Aplica transformações aleatórias para aumentar
    diversidade dos dados de treinamento.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        apply_prob: float = 0.5,
        pitch_shift_range: Tuple[float, float] = (-2.0, 2.0),
        speed_change_range: Tuple[float, float] = (0.9, 1.1),
        noise_level_range: Tuple[float, float] = (0.0, 0.05),
        volume_change_range: Tuple[float, float] = (0.8, 1.2),
        spec_augment_prob: float = 0.3,
        time_mask_num: int = 2,
        freq_mask_num: int = 2,
        time_mask_width: int = 25,
        freq_mask_width: int = 15,
    ):
        """
        Inicializa sistema de augmentação.

        Args:
            sample_rate: Taxa de amostragem
            apply_prob: Probabilidade de aplicar augmentação
            pitch_shift_range: Range de pitch shift em semitons
            speed_change_range: Range de mudança de velocidade
            noise_level_range: Range de nível de ruído
            volume_change_range: Range de mudança de volume
            spec_augment_prob: Probabilidade de SpecAugment
            time_mask_num: Número de máscaras temporais
            freq_mask_num: Número de máscaras de frequência
            time_mask_width: Largura máxima da máscara temporal
            freq_mask_width: Largura máxima da máscara de frequência
        """
        self.sample_rate = sample_rate
        self.apply_prob = apply_prob
        self.pitch_shift_range = pitch_shift_range
        self.speed_change_range = speed_change_range
        self.noise_level_range = noise_level_range
        self.volume_change_range = volume_change_range
        self.spec_augment_prob = spec_augment_prob
        self.time_mask_num = time_mask_num
        self.freq_mask_num = freq_mask_num
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width

    def should_apply_augmentation(self) -> bool:
        """Decide se deve aplicar augmentação."""
        return random.random() < self.apply_prob

    def pitch_shift(
        self,
        waveform: torch.Tensor,
        shift_semitones: Optional[float] = None
    ) -> torch.Tensor:
        """
        Aplica pitch shifting ao áudio.

        Args:
            waveform: Waveform de entrada [length]
            shift_semitones: Shift em semitons (None = aleatório)

        Returns:
            Waveform com pitch modificado
        """
        if shift_semitones is None:
            shift_semitones = random.uniform(*self.pitch_shift_range)

        if abs(shift_semitones) < 0.1:  # Skip se muito pequeno
            return waveform

        # Simular pitch shift mudando velocidade (aproximação)
        # Em produção, usar biblioteca especializada como librosa
        pitch_ratio = 2 ** (shift_semitones / 12.0)
        new_length = max(1, int(len(waveform) / pitch_ratio))

        # Usar resample simples
        if new_length == len(waveform):
            return waveform

        # Redimensionar usando interpolação
        resized = F.interpolate(
            waveform.view(1, 1, -1),
            size=new_length,
            mode='linear',
            align_corners=False
        ).squeeze()

        return resized

    def speed_change(
        self,
        waveform: torch.Tensor,
        speed_factor: Optional[float] = None
    ) -> torch.Tensor:
        """
        Modifica velocidade do áudio.

        Args:
            waveform: Waveform de entrada [length]
            speed_factor: Fator de velocidade (None = aleatório)

        Returns:
            Waveform com velocidade modificada
        """
        if speed_factor is None:
            speed_factor = random.uniform(*self.speed_change_range)

        if abs(speed_factor - 1.0) < 0.01:  # Skip se muito pequeno
            return waveform

        # Resample para mudar velocidade
        new_length = max(1, int(len(waveform) / speed_factor))

        if new_length == len(waveform):
            return waveform

        # Redimensionar usando interpolação
        resampled = F.interpolate(
            waveform.view(1, 1, -1),
            size=new_length,
            mode='linear',
            align_corners=False
        ).squeeze()

        return resampled

    def add_noise(
        self,
        waveform: torch.Tensor,
        noise_level: Optional[float] = None
    ) -> torch.Tensor:
        """
        Adiciona ruído gaussiano ao áudio.

        Args:
            waveform: Waveform de entrada [length]
            noise_level: Nível de ruído (None = aleatório)

        Returns:
            Waveform com ruído adicionado
        """
        if noise_level is None:
            noise_level = random.uniform(*self.noise_level_range)

        if noise_level < 1e-6:  # Skip se muito pequeno
            return waveform

        # Gerar ruído gaussiano
        noise = torch.randn_like(waveform) * noise_level

        # Adicionar ruído
        noisy = waveform + noise

        # Normalizar para evitar clipping
        max_val = torch.max(torch.abs(noisy))
        if max_val > 1.0:
            noisy = noisy / max_val

        return noisy

    def volume_change(
        self,
        waveform: torch.Tensor,
        volume_factor: Optional[float] = None
    ) -> torch.Tensor:
        """
        Modifica volume do áudio.

        Args:
            waveform: Waveform de entrada [length]
            volume_factor: Fator de volume (None = aleatório)

        Returns:
            Waveform com volume modificado
        """
        if volume_factor is None:
            volume_factor = random.uniform(*self.volume_change_range)

        return waveform * volume_factor

    def spec_augment(
        self,
        mel_spec: torch.Tensor,
        apply_time_mask: bool = True,
        apply_freq_mask: bool = True
    ) -> torch.Tensor:
        """
        Aplica SpecAugment ao mel-spectrogram.

        Args:
            mel_spec: Mel-spectrogram [n_mels, time_frames]
            apply_time_mask: Se deve aplicar máscaras temporais
            apply_freq_mask: Se deve aplicar máscaras de frequência

        Returns:
            Mel-spectrogram augmentado
        """
        augmented = mel_spec.clone()
        n_mels, time_frames = augmented.shape

        # Time masking
        if apply_time_mask and time_frames > self.time_mask_width:
            for _ in range(self.time_mask_num):
                mask_width = random.randint(1, min(self.time_mask_width, time_frames // 4))
                mask_start = random.randint(0, time_frames - mask_width)
                augmented[:, mask_start:mask_start + mask_width] = 0

        # Frequency masking
        if apply_freq_mask and n_mels > self.freq_mask_width:
            for _ in range(self.freq_mask_num):
                mask_width = random.randint(1, min(self.freq_mask_width, n_mels // 4))
                mask_start = random.randint(0, n_mels - mask_width)
                augmented[mask_start:mask_start + mask_width, :] = 0

        return augmented

    def random_crop_or_pad(
        self,
        waveform: torch.Tensor,
        target_length: int,
        pad_mode: str = "constant"
    ) -> torch.Tensor:
        """
        Faz crop ou padding aleatório do áudio.

        Args:
            waveform: Waveform de entrada [length]
            target_length: Comprimento alvo
            pad_mode: Modo de padding ("constant", "reflect", "replicate")

        Returns:
            Waveform com comprimento alvo
        """
        current_length = len(waveform)

        if current_length == target_length:
            return waveform
        elif current_length > target_length:
            # Crop aleatório
            start_idx = random.randint(0, current_length - target_length)
            return waveform[start_idx:start_idx + target_length]
        else:
            # Padding
            pad_length = target_length - current_length

            if pad_mode == "constant":
                # Padding com zeros
                return F.pad(waveform, (0, pad_length), value=0)
            elif pad_mode == "reflect":
                # Padding reflexivo
                return F.pad(waveform, (0, pad_length), mode="reflect")
            elif pad_mode == "replicate":
                # Padding replicando bordas
                return F.pad(waveform, (0, pad_length), mode="replicate")
            else:
                raise ValueError(f"Modo de padding não suportado: {pad_mode}")

    def augment_waveform(
        self,
        waveform: torch.Tensor,
        augment_types: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Aplica augmentação completa ao waveform.

        Args:
            waveform: Waveform de entrada [length]
            augment_types: Tipos de augmentação (None = todas)

        Returns:
            Waveform augmentado
        """
        if not self.should_apply_augmentation():
            return waveform

        available_types = ["pitch", "speed", "noise", "volume"]
        if augment_types is None:
            augment_types = available_types

        augmented = waveform.clone()

        # Aplicar augmentações em ordem aleatória
        random.shuffle(augment_types)

        for aug_type in augment_types:
            if aug_type == "pitch" and random.random() < 0.5:
                augmented = self.pitch_shift(augmented)
            elif aug_type == "speed" and random.random() < 0.5:
                augmented = self.speed_change(augmented)
            elif aug_type == "noise" and random.random() < 0.3:
                augmented = self.add_noise(augmented)
            elif aug_type == "volume" and random.random() < 0.5:
                augmented = self.volume_change(augmented)

        return augmented

    def augment_mel_spectrogram(
        self,
        mel_spec: torch.Tensor
    ) -> torch.Tensor:
        """
        Aplica augmentação ao mel-spectrogram.

        Args:
            mel_spec: Mel-spectrogram [n_mels, time_frames]

        Returns:
            Mel-spectrogram augmentado
        """
        if random.random() < self.spec_augment_prob:
            return self.spec_augment(mel_spec)
        return mel_spec

    def augment_batch(
        self,
        batch: dict,
        augment_audio: bool = True,
        augment_mel: bool = True
    ) -> dict:
        """
        Aplica augmentação a um batch inteiro.

        Args:
            batch: Batch de dados
            augment_audio: Se deve augmentar waveforms
            augment_mel: Se deve augmentar mel-spectrograms

        Returns:
            Batch augmentado
        """
        augmented_batch = batch.copy()

        if augment_audio and 'audio' in batch:
            batch_size = batch['audio'].size(0)
            augmented_audio = []

            for i in range(batch_size):
                audio = batch['audio'][i]
                # Trim para remover padding
                if 'audio_lengths' in batch:
                    audio = audio[:batch['audio_lengths'][i]]

                augmented = self.augment_waveform(audio)
                augmented_audio.append(augmented)

            # Re-aplicar padding se necessário
            if len(set(len(a) for a in augmented_audio)) > 1:
                max_len = max(len(a) for a in augmented_audio)
                padded_audio = torch.zeros(batch_size, max_len)
                for i, audio in enumerate(augmented_audio):
                    padded_audio[i, :len(audio)] = audio
                augmented_batch['audio'] = padded_audio
            else:
                augmented_batch['audio'] = torch.stack(augmented_audio)

        if augment_mel and 'mel' in batch:
            batch_size = batch['mel'].size(0)
            augmented_mel = []

            for i in range(batch_size):
                mel = batch['mel'][i]
                # Trim para remover padding
                if 'mel_lengths' in batch:
                    mel = mel[:, :batch['mel_lengths'][i]]

                augmented = self.augment_mel_spectrogram(mel)
                augmented_mel.append(augmented)

            # Re-aplicar padding se necessário
            if len(set(m.size(1) for m in augmented_mel)) > 1:
                max_len = max(m.size(1) for m in augmented_mel)
                n_mels = augmented_mel[0].size(0)
                padded_mel = torch.zeros(batch_size, n_mels, max_len)
                for i, mel in enumerate(augmented_mel):
                    padded_mel[i, :, :mel.size(1)] = mel
                augmented_batch['mel'] = padded_mel
            else:
                augmented_batch['mel'] = torch.stack(augmented_mel)

        return augmented_batch

    def get_config(self) -> dict:
        """
        Retorna configuração do augmentador.

        Returns:
            Dict com parâmetros de configuração
        """
        return {
            'sample_rate': self.sample_rate,
            'apply_prob': self.apply_prob,
            'pitch_shift_range': self.pitch_shift_range,
            'speed_change_range': self.speed_change_range,
            'noise_level_range': self.noise_level_range,
            'volume_change_range': self.volume_change_range,
            'spec_augment_prob': self.spec_augment_prob,
            'time_mask_num': self.time_mask_num,
            'freq_mask_num': self.freq_mask_num,
            'time_mask_width': self.time_mask_width,
            'freq_mask_width': self.freq_mask_width,
        }
