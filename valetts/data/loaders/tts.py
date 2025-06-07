"""
Data loader para datasets TTS.

Suporta múltiplos formatos de dataset:
- LJSpeech
- VCTK
- Custom formats
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import random

from valetts.data.preprocessing.audio import AudioPreprocessor
from valetts.data.preprocessing.text import TextPreprocessor


class TTSDataset(Dataset):
    """
    Dataset para dados TTS.

    Carrega e processa áudio + texto para treinamento
    de modelos text-to-speech.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        metadata_file: Union[str, Path],
        audio_preprocessor: AudioPreprocessor,
        text_preprocessor: TextPreprocessor,
        dataset_format: str = "ljspeech",
        max_audio_length: Optional[int] = None,
        max_text_length: Optional[int] = None,
        min_audio_length: Optional[int] = None,
        min_text_length: Optional[int] = None,
        speaker_embedding_dim: Optional[int] = None,
        cache_audio: bool = False,
        filter_by_length: bool = True,
    ):
        """
        Inicializa dataset TTS.

        Args:
            data_dir: Diretório com arquivos de áudio
            metadata_file: Arquivo com metadados (texto, speakers, etc.)
            audio_preprocessor: Preprocessador de áudio
            text_preprocessor: Preprocessador de texto
            dataset_format: Formato do dataset (ljspeech, vctk, custom)
            max_audio_length: Comprimento máximo de áudio (samples)
            max_text_length: Comprimento máximo de texto (chars)
            min_audio_length: Comprimento mínimo de áudio (samples)
            min_text_length: Comprimento mínimo de texto (chars)
            speaker_embedding_dim: Dimensão de embeddings de speaker
            cache_audio: Se deve cachear áudio processado
            filter_by_length: Se deve filtrar por comprimento
        """
        self.data_dir = Path(data_dir)
        self.metadata_file = Path(metadata_file)
        self.audio_preprocessor = audio_preprocessor
        self.text_preprocessor = text_preprocessor
        self.dataset_format = dataset_format.lower()
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.min_audio_length = min_audio_length or 1000
        self.min_text_length = min_text_length or 5
        self.speaker_embedding_dim = speaker_embedding_dim
        self.cache_audio = cache_audio
        self.filter_by_length = filter_by_length

        # Cache para áudio processado
        self.audio_cache = {} if cache_audio else None

        # Carregar metadados
        self.metadata = self._load_metadata()

        # Filtrar dados se configurado
        if filter_by_length:
            self.metadata = self._filter_by_length()

        # Mapear speakers
        self.speaker_to_id = self._build_speaker_mapping()

        print(f"📊 Dataset carregado: {len(self.metadata)} amostras")
        if self.speaker_to_id:
            print(f"🎭 Speakers encontrados: {len(self.speaker_to_id)}")

    def _load_metadata(self) -> pd.DataFrame:
        """Carrega metadados baseado no formato do dataset."""
        if self.dataset_format == "ljspeech":
            return self._load_ljspeech_metadata()
        elif self.dataset_format == "vctk":
            return self._load_vctk_metadata()
        elif self.dataset_format == "custom":
            return self._load_custom_metadata()
        else:
            raise ValueError(f"Formato de dataset não suportado: {self.dataset_format}")

    def _load_ljspeech_metadata(self) -> pd.DataFrame:
        """Carrega metadados do LJSpeech."""
        # LJSpeech usa formato: filename|text
        data = []

        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        filename = parts[0] + '.wav'
                        text = '|'.join(parts[1:])  # Juntar caso tenha | no texto

                        audio_path = self.data_dir / 'wavs' / filename
                        if audio_path.exists():
                            data.append({
                                'audio_path': str(audio_path),
                                'text': text,
                                'speaker_id': 'ljspeech',  # Single speaker
                                'filename': filename
                            })

        return pd.DataFrame(data)

    def _load_vctk_metadata(self) -> pd.DataFrame:
        """Carrega metadados do VCTK."""
        # VCTK: txt files com mesmo nome dos wavs
        data = []

        # Procurar arquivos .wav
        wav_files = list(self.data_dir.rglob('*.wav'))

        for wav_path in wav_files:
            # Encontrar arquivo de texto correspondente
            txt_path = wav_path.with_suffix('.txt')

            if txt_path.exists():
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()

                # Extrair speaker ID do nome do arquivo (ex: p225_001.wav -> p225)
                speaker_id = wav_path.stem.split('_')[0]

                data.append({
                    'audio_path': str(wav_path),
                    'text': text,
                    'speaker_id': speaker_id,
                    'filename': wav_path.name
                })

        return pd.DataFrame(data)

    def _load_custom_metadata(self) -> pd.DataFrame:
        """Carrega metadados de formato customizado (JSON ou CSV)."""
        if self.metadata_file.suffix == '.json':
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        elif self.metadata_file.suffix == '.csv':
            return pd.read_csv(self.metadata_file)
        else:
            raise ValueError(f"Formato de metadata não suportado: {self.metadata_file.suffix}")

    def _filter_by_length(self) -> pd.DataFrame:
        """Filtra amostras por comprimento."""
        filtered_data = []

        for idx, row in self.metadata.iterrows():
            # Verificar comprimento do texto
            text_len = len(row['text'])
            if text_len < self.min_text_length:
                continue
            if self.max_text_length and text_len > self.max_text_length:
                continue

            filtered_data.append(row)

        print(f"🔍 Filtrado: {len(self.metadata)} → {len(filtered_data)} amostras")
        return pd.DataFrame(filtered_data)

    def _build_speaker_mapping(self) -> Dict[str, int]:
        """Constrói mapeamento speaker_id → int."""
        if 'speaker_id' not in self.metadata.columns:
            return {}

        unique_speakers = sorted(self.metadata['speaker_id'].unique())
        return {speaker: idx for idx, speaker in enumerate(unique_speakers)}

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retorna uma amostra do dataset.

        Args:
            idx: Índice da amostra

        Returns:
            Dict com tensors processados
        """
        row = self.metadata.iloc[idx]

        # Carregar e processar áudio
        audio_path = row['audio_path']

        try:
            audio = self.audio_preprocessor.load_audio(audio_path)
            audio = self.audio_preprocessor.trim_silence(audio)
        except Exception as e:
            print(f"⚠️ Erro carregando áudio {audio_path}: {e}")
            # Retornar silêncio como fallback
            audio = torch.zeros(self.audio_preprocessor.sample_rate)

        # Processar mel-spectrogram
        mel_spec = self.audio_preprocessor.mel_spectrogram(audio)

        # Processar texto
        text = row['text']
        text_ids = self.text_preprocessor.encode(text)

        # Preparar dados de retorno
        sample = {
            'audio': audio,
            'mel': mel_spec,
            'text': torch.tensor(text_ids, dtype=torch.long),
            'text_raw': text,
            'audio_path': audio_path,
        }

        # Adicionar speaker embedding se disponível
        if self.speaker_to_id and 'speaker_id' in row:
            speaker_id = self.speaker_to_id[row['speaker_id']]
            sample['speaker_id'] = torch.tensor(speaker_id, dtype=torch.long)

        return sample

    def get_speaker_count(self) -> int:
        """Retorna número de speakers únicos."""
        return len(self.speaker_to_id)

    def get_sample_speaker_ids(self) -> List[str]:
        """Retorna lista de speaker IDs."""
        return list(self.speaker_to_id.keys())


class TTSDataLoader:
    """
    Data loader para datasets TTS com collate function otimizada.
    """

    def __init__(
        self,
        dataset: TTSDataset,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
        max_batch_audio_length: Optional[int] = None,
        max_batch_text_length: Optional[int] = None,
    ):
        """
        Inicializa data loader TTS.

        Args:
            dataset: Dataset TTS
            batch_size: Tamanho do batch
            shuffle: Se deve embaralhar dados
            num_workers: Número de workers para loading
            pin_memory: Se deve usar pin memory
            drop_last: Se deve descartar último batch incompleto
            max_batch_audio_length: Comprimento máximo de áudio no batch
            max_batch_text_length: Comprimento máximo de texto no batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_batch_audio_length = max_batch_audio_length
        self.max_batch_text_length = max_batch_text_length

        # Criar DataLoader do PyTorch
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate function para criar batch com padding.

        Args:
            batch: Lista de amostras

        Returns:
            Batch processado com padding
        """
        # Separar componentes
        audios = [item['audio'] for item in batch]
        mels = [item['mel'] for item in batch]
        texts = [item['text'] for item in batch]

        # Calcular comprimentos
        audio_lengths = [len(audio) for audio in audios]
        mel_lengths = [mel.size(1) for mel in mels]
        text_lengths = [len(text) for text in texts]

        # Aplicar limites de batch se especificados
        if self.max_batch_audio_length:
            max_audio_len = min(max(audio_lengths), self.max_batch_audio_length)
        else:
            max_audio_len = max(audio_lengths)

        if self.max_batch_text_length:
            max_text_len = min(max(text_lengths), self.max_batch_text_length)
        else:
            max_text_len = max(text_lengths)

        max_mel_len = max(mel_lengths)

        # Aplicar padding
        batch_size = len(batch)

        # Padding para áudio
        audio_batch = torch.zeros(batch_size, max_audio_len)
        for i, audio in enumerate(audios):
            length = min(len(audio), max_audio_len)
            audio_batch[i, :length] = audio[:length]

        # Padding para mel-spectrograms
        n_mels = mels[0].size(0)
        mel_batch = torch.zeros(batch_size, n_mels, max_mel_len)
        for i, mel in enumerate(mels):
            length = min(mel.size(1), max_mel_len)
            mel_batch[i, :, :length] = mel[:, :length]

        # Padding para texto
        text_batch = torch.zeros(batch_size, max_text_len, dtype=torch.long)
        pad_id = self.dataset.text_preprocessor.vocab[self.dataset.text_preprocessor.pad_token]
        text_batch.fill_(pad_id)

        for i, text in enumerate(texts):
            length = min(len(text), max_text_len)
            text_batch[i, :length] = text[:length]

        # Preparar batch final
        result = {
            'audio': audio_batch,
            'audio_lengths': torch.tensor(audio_lengths),
            'mel': mel_batch,
            'mel_lengths': torch.tensor(mel_lengths),
            'text': text_batch,
            'text_lengths': torch.tensor(text_lengths),
        }

        # Adicionar speaker IDs se disponível
        if 'speaker_id' in batch[0]:
            speaker_ids = [item['speaker_id'] for item in batch]
            result['speaker_ids'] = torch.stack(speaker_ids)

        # Adicionar metadados úteis
        result['text_raw'] = [item['text_raw'] for item in batch]
        result['audio_paths'] = [item['audio_path'] for item in batch]

        return result

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def get_dataset_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o dataset."""
        return {
            'num_samples': len(self.dataset),
            'num_speakers': self.dataset.get_speaker_count(),
            'speaker_ids': self.dataset.get_sample_speaker_ids(),
            'vocab_size': self.dataset.text_preprocessor.get_vocab_size(),
            'sample_rate': self.dataset.audio_preprocessor.sample_rate,
            'n_mels': self.dataset.audio_preprocessor.n_mels,
        }


def create_tts_dataloader(
    data_dir: Union[str, Path],
    metadata_file: Union[str, Path],
    dataset_format: str = "ljspeech",
    batch_size: int = 16,
    sample_rate: int = 22050,
    n_mels: int = 80,
    vocab_size: int = 512,
    max_audio_length: Optional[int] = None,
    max_text_length: Optional[int] = None,
    cache_audio: bool = False,
    num_workers: int = 4,
    **kwargs
) -> TTSDataLoader:
    """
    Função helper para criar data loader TTS rapidamente.

    Args:
        data_dir: Diretório de dados
        metadata_file: Arquivo de metadados
        dataset_format: Formato do dataset
        batch_size: Tamanho do batch
        sample_rate: Taxa de amostragem
        n_mels: Número de canais mel
        vocab_size: Tamanho do vocabulário
        max_audio_length: Comprimento máximo de áudio
        max_text_length: Comprimento máximo de texto
        cache_audio: Se deve cachear áudio
        num_workers: Número de workers
        **kwargs: Argumentos adicionais

    Returns:
        Data loader configurado
    """
    # Criar preprocessadores
    audio_preprocessor = AudioPreprocessor(
        sample_rate=sample_rate,
        n_mels=n_mels,
        **{k: v for k, v in kwargs.items() if k.startswith('audio_')}
    )

    text_preprocessor = TextPreprocessor(
        vocab_size=vocab_size,
        **{k: v for k, v in kwargs.items() if k.startswith('text_')}
    )

    # Criar dataset
    dataset = TTSDataset(
        data_dir=data_dir,
        metadata_file=metadata_file,
        audio_preprocessor=audio_preprocessor,
        text_preprocessor=text_preprocessor,
        dataset_format=dataset_format,
        max_audio_length=max_audio_length,
        max_text_length=max_text_length,
        cache_audio=cache_audio,
    )

    # Criar data loader
    dataloader = TTSDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        **{k: v for k, v in kwargs.items() if k.startswith('loader_')}
    )

    return dataloader
