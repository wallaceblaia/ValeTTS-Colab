"""
Utilitários de processamento de áudio para o ValeTTS.
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchaudio


class AudioProcessor:
    """Processador de áudio para TTS."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mel: int = 80,
        mel_fmin: float = 0.0,
        mel_fmax: Optional[float] = None,
    ):
        """
        Inicializa o processador de áudio.

        Args:
            sample_rate: Taxa de amostragem do áudio
            n_fft: Tamanho da FFT
            hop_length: Tamanho do passo
            win_length: Tamanho da janela
            n_mel: Número de filtros mel
            mel_fmin: Frequência mínima mel
            mel_fmax: Frequência máxima mel
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel = n_mel
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax or sample_rate // 2

        # Transformações de mel
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mel,
            f_min=mel_fmin,
            f_max=self.mel_fmax,
        )

    def load_audio(self, path: str) -> Tuple[torch.Tensor, int]:
        """
        Carrega um arquivo de áudio.

        Args:
            path: Caminho para o arquivo de áudio

        Returns:
            Tuple com o tensor de áudio e a taxa de amostragem
        """
        audio, sr = torchaudio.load(path)

        # Converter para mono se necessário
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Reamostrar se necessário
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)

        return audio.squeeze(0), self.sample_rate

    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Converte áudio para espectrograma mel.

        Args:
            audio: Tensor de áudio

        Returns:
            Espectrograma mel
        """
        mel = self.mel_transform(audio)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel

    def normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Normaliza o áudio.

        Args:
            audio: Tensor de áudio

        Returns:
            Áudio normalizado
        """
        return audio / torch.max(torch.abs(audio))

    def trim_silence(
        self, audio: torch.Tensor, threshold: float = 0.01
    ) -> torch.Tensor:
        """
        Remove silêncio do início e fim do áudio.

        Args:
            audio: Tensor de áudio
            threshold: Limiar para detecção de silêncio

        Returns:
            Áudio com silêncio removido
        """
        # Encontrar índices onde o áudio está acima do limiar
        non_silent = torch.abs(audio) > threshold
        indices = torch.where(non_silent)[0]

        if len(indices) == 0:
            return audio

        start = indices[0]
        end = indices[-1] + 1

        return audio[start:end]
