"""
Dataset TTS para treinamento
===========================

Dataset que carrega metadados JSON/CSV e áudios para treinamento VITS2.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from ..preprocessing.audio import AudioPreprocessor
from ..preprocessing.text import TextPreprocessor

logger = logging.getLogger(__name__)


class TTSDataset(Dataset):
    """
    Dataset TTS compatível com metadados JSON/CSV gerados pelo DatasetBuilder.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        text_config: Optional[Dict[str, Any]] = None,
        audio_config: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Inicializa dataset TTS.

        Args:
            data_dir: Diretório do dataset
            split: Split a usar (train/val/test)
            text_config: Configuração do processador de texto
            audio_config: Configuração do processador de áudio
            max_samples: Máximo de amostras (para debug)
        """
        self.data_dir = Path(data_dir)
        self.split = split

        # Processadores
        self.text_processor = TextPreprocessor(**(text_config or {}))

        # AudioPreprocessor com configuração padrão
        audio_cfg = audio_config or {}
        self.audio_processor = AudioPreprocessor(
            sample_rate=audio_cfg.get("sample_rate", 22050),
            n_fft=audio_cfg.get("n_fft", 1024),
            hop_length=audio_cfg.get("hop_length", 256),
            n_mels=audio_cfg.get("n_mels", 80),
        )

        # Carregar metadados
        self.samples = self._load_metadata()

        # Limitar amostras se necessário (para debug)
        if max_samples:
            self.samples = self.samples[:max_samples]
            logger.info(f"🐛 DEBUG: Limitado a {max_samples} amostras")

        # Estatísticas
        self.speakers = list(
            set(sample["speaker_id"] for sample in self.samples)
        )
        self.speaker_to_id = {
            speaker: i for i, speaker in enumerate(self.speakers)
        }

        logger.info(f"📊 TTSDataset criado - Split: {split}")
        logger.info(f"   📈 Amostras: {len(self.samples)}")
        logger.info(f"   🎙️ Falantes: {len(self.speakers)} ({self.speakers})")

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Carrega metadados do split específico."""
        # Tentar carregar split específico primeiro
        split_file = self.data_dir / f"{self.split}_split.json"

        if split_file.exists():
            logger.info(f"📋 Carregando split: {split_file}")
            with open(split_file, "r", encoding="utf-8") as f:
                return json.load(f)

        # Fallback para metadata completo
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            logger.info(f"📋 Carregando metadata completo: {metadata_file}")
            with open(metadata_file, "r", encoding="utf-8") as f:
                all_samples = json.load(f)

            # Filtrar por split se disponível
            filtered = [s for s in all_samples if s.get("split") == self.split]
            if filtered:
                return filtered

            # Se não há splits, usar tudo (modo debug)
            logger.warning(
                f"⚠️ Split '{self.split}' não encontrado, usando todas as amostras"
            )
            return all_samples

        # Fallback para CSV com split automático
        csv_file = self.data_dir / "metadata.csv"
        if csv_file.exists():
            logger.info(f"📋 Carregando CSV: {csv_file}")
            df = pd.read_csv(csv_file)

            # Se há coluna split, usar ela
            if "split" in df.columns:
                df_split = df[df["split"] == self.split]
                if len(df_split) > 0:
                    logger.info(
                        f"📊 Split {self.split}: {len(df_split)} amostras"
                    )
                    return df_split.to_dict("records")

            # SPLIT AUTOMÁTICO se não há coluna split
            logger.info(f"🔀 Criando split automático para {self.split}")

            # Shuffle determinístico baseado no speaker para manter consistência
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

            total_samples = len(df)
            if self.split == "train":
                # 90% para treino
                end_idx = int(0.9 * total_samples)
                df_split = df[:end_idx]
            elif self.split == "val":
                # 10% para validação
                start_idx = int(0.9 * total_samples)
                df_split = df[start_idx:]
            else:
                # Split desconhecido, usar tudo
                df_split = df

            logger.info(
                f"📊 Split automático {self.split}: {len(df_split)}/{total_samples} amostras"
            )
            return df_split.to_dict("records")

        raise FileNotFoundError(
            f"❌ Nenhum arquivo de metadados encontrado em {self.data_dir}"
        )

    def __len__(self) -> int:
        """Retorna número de amostras."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retorna uma amostra do dataset.

        Returns:
            Dict com keys: text, audio, speaker_id, text_length, audio_length
        """
        sample = self.samples[idx]

        # Processar texto
        text = sample["text"]
        text_tokens = self.text_processor.encode(text)
        text_tensor = torch.LongTensor(text_tokens)

        # Carregar e processar áudio
        audio_path = self.data_dir / sample["audio_path"]
        if not audio_path.exists():
            # Tentar path relativo ao data_dir
            audio_path = Path(sample["audio_path"])
            if not audio_path.is_absolute():
                audio_path = self.data_dir / audio_path

        # Carregar áudio
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))

            # Processar áudio (resample se necessário)
            if sample_rate != self.audio_processor.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, self.audio_processor.sample_rate
                )
                waveform = resampler(waveform)

            # Converter para mono se necessário
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            else:
                waveform = waveform.squeeze(0)

            # Gerar mel-spectrogram
            mel_spec = self.audio_processor.mel_spectrogram(waveform)

            # Speaker ID
            speaker_id = self.speaker_to_id[sample["speaker_id"]]

            return {
                "text": text_tensor,
                "audio": waveform.squeeze(0),  # Remove batch dim
                "mel": mel_spec.squeeze(0),  # Remove batch dim
                "speaker_id": torch.LongTensor([speaker_id]),
                "text_length": torch.LongTensor([len(text_tokens)]),
                "audio_length": torch.LongTensor([waveform.shape[-1]]),
                "sample_id": sample.get("sample_id", f"sample_{idx}"),
            }

        except Exception as e:
            logger.error(f"❌ Erro ao carregar {audio_path}: {e}")
            # Retornar dummy data para não quebrar o treinamento
            return self._get_dummy_sample(idx)

    def _get_dummy_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retorna amostra dummy em caso de erro."""
        return {
            "text": torch.LongTensor([1, 2, 3]),  # Dummy tokens
            "audio": torch.zeros(16000),  # 1 segundo de silêncio
            "mel": torch.zeros(80, 100),  # Dummy mel-spec
            "speaker_id": torch.LongTensor([0]),
            "text_length": torch.LongTensor([3]),
            "audio_length": torch.LongTensor([16000]),
            "sample_id": f"dummy_{idx}",
        }

    def get_speaker_info(self) -> Dict[str, Any]:
        """Retorna informações dos falantes."""
        return {
            "num_speakers": len(self.speakers),
            "speakers": self.speakers,
            "speaker_to_id": self.speaker_to_id,
        }


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Collate function para o DataLoader.

    Args:
        batch: Lista de amostras do dataset

    Returns:
        Batch padded e pronto para treinamento
    """
    # Separar componentes
    texts = [item["text"] for item in batch]
    audios = [item["audio"] for item in batch]
    mels = [item["mel"] for item in batch]
    speaker_ids = torch.cat([item["speaker_id"] for item in batch])
    text_lengths = torch.cat([item["text_length"] for item in batch])
    audio_lengths = torch.cat([item["audio_length"] for item in batch])

    # Pad sequences
    text_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)
    audio_padded = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)

    # Pad mel-spectrograms
    max_mel_len = max(mel.shape[-1] for mel in mels)
    mel_padded = torch.zeros(len(mels), mels[0].shape[0], max_mel_len)
    mel_lengths = []
    for i, mel in enumerate(mels):
        mel_padded[i, :, : mel.shape[-1]] = mel
        mel_lengths.append(mel.shape[-1])

    mel_lengths = torch.LongTensor(mel_lengths)

    return {
        "text": text_padded,
        "audio": audio_padded,
        "mel": mel_padded,
        "speaker_ids": speaker_ids,
        "text_lengths": text_lengths,
        "audio_lengths": audio_lengths,
        "mel_lengths": mel_lengths,
    }
