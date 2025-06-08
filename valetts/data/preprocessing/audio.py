"""
Preprocessamento de áudio para ValeTTS.

Contém funcionalidades para:
- Conversão áudio → mel-spectrogram
- STFT e processamento de frequência
- Normalização e padronização
- Suporte a múltiplas frequências de amostragem
"""

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Optional, Tuple, Union
import librosa
from pathlib import Path


class AudioPreprocessor:
    """
    Preprocessador de áudio para modelos TTS.

    Converte waveforms para mel-spectrograms e outras representações
    necessárias para treinamento e inferência.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        normalize: bool = True,
        preemphasis: float = 0.97,
        ref_level_db: float = 20.0,
        min_level_db: float = -100.0,
    ):
        """
        Inicializa o preprocessador de áudio.

        Args:
            sample_rate: Taxa de amostragem de áudio
            n_fft: Tamanho da FFT
            hop_length: Hop length para STFT
            win_length: Tamanho da janela (None = n_fft)
            n_mels: Número de canais mel
            fmin: Frequência mínima para mel scale
            fmax: Frequência máxima para mel scale (None = sr/2)
            normalize: Se deve normalizar mel-spectrograms
            preemphasis: Coeficiente de preemphasis
            ref_level_db: Nível de referência em dB
            min_level_db: Nível mínimo em dB
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        self.normalize = normalize
        self.preemphasis = preemphasis
        self.ref_level_db = ref_level_db
        self.min_level_db = min_level_db

        # Criar mel filter bank
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=self.fmax
        )
        self.mel_basis = torch.from_numpy(self.mel_basis).float()

    def load_audio(
        self,
        path: Union[str, Path],
        target_sr: Optional[int] = None
    ) -> torch.Tensor:
        """
        Carrega arquivo de áudio.

        Args:
            path: Caminho para o arquivo de áudio
            target_sr: Taxa de amostragem alvo (None = usar self.sample_rate)

        Returns:
            Waveform normalizado [length]
        """
        target_sr = target_sr or self.sample_rate

        # Carregar áudio com torchaudio
        waveform, sr = torchaudio.load(str(path))

        # Converter para mono se necessário
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        else:
            waveform = waveform.squeeze(0)

        # Resample se necessário
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)

        # Normalizar
        waveform = waveform / torch.max(torch.abs(waveform))

        return waveform

    def mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Converte waveform para mel-spectrogram.

        Args:
            waveform: Waveform de entrada [length]

        Returns:
            Mel-spectrogram [n_mels, time_frames]
        """
        # Calcular STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length),
            return_complex=True
        )

        # Magnitude spectrogram
        magnitude = torch.abs(stft)

        # Converter para mel scale
        mel_basis = self.mel_basis.to(magnitude.device)
        mel_spec = torch.matmul(mel_basis, magnitude)

        # Converter para log scale
        mel_spec = torch.clamp(mel_spec, min=1e-5)
        mel_spec = torch.log(mel_spec)

        return mel_spec

    def process_batch(
        self,
        waveforms: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processa batch de waveforms.

        Args:
            waveforms: Batch de waveforms [batch_size, max_length]
            lengths: Comprimentos reais de cada waveform [batch_size]

        Returns:
            mel_specs: Mel-spectrograms [batch_size, n_mels, max_time_frames]
            mel_lengths: Comprimentos dos mel-spectrograms [batch_size]
        """
        batch_size = waveforms.size(0)
        mel_specs = []
        mel_lengths = []

        for i in range(batch_size):
            # Extrair waveform individual
            if lengths is not None:
                waveform = waveforms[i, :lengths[i]]
            else:
                waveform = waveforms[i]

            # Processar
            mel_spec = self.mel_spectrogram(waveform)
            mel_specs.append(mel_spec)
            mel_lengths.append(mel_spec.size(1))

        # Fazer padding para batch
        max_mel_len = max(mel_lengths)
        mel_batch = torch.zeros(batch_size, self.n_mels, max_mel_len)

        for i, mel_spec in enumerate(mel_specs):
            mel_batch[i, :, :mel_spec.size(1)] = mel_spec

        return mel_batch, torch.tensor(mel_lengths)

    def trim_silence(
        self,
        waveform: torch.Tensor,
        threshold: float = 0.01,
        frame_length: int = 1024,
        hop_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Remove silêncio do início e fim do áudio.

        Args:
            waveform: Waveform de entrada [length]
            threshold: Threshold para detectar silêncio
            frame_length: Tamanho do frame para análise
            hop_length: Hop length para análise

        Returns:
            Waveform sem silêncio
        """
        hop_length = hop_length or frame_length // 4

        # Calcular energia por frame
        energy = []
        for i in range(0, len(waveform) - frame_length, hop_length):
            frame = waveform[i:i + frame_length]
            energy.append(torch.mean(frame ** 2))

        if not energy:
            return waveform

        energy = torch.stack(energy)

        # Encontrar início e fim não-silenciosos
        non_silent = energy > threshold
        if not torch.any(non_silent):
            return waveform  # Não remover se tudo for silêncio

        start_idx = torch.nonzero(non_silent)[0].item() * hop_length
        end_idx = (torch.nonzero(non_silent)[-1].item() + 1) * hop_length
        end_idx = min(end_idx + frame_length, len(waveform))

        return waveform[start_idx:end_idx]




