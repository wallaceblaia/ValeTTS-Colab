"""
Data loader específico para datasets TTS em inglês.

Suporta o formato ValeTTS com processamento específico para inglês:
- Phonemes G2P para inglês
- Normalização de texto específica
- Mapeamento correto de locale
"""

import torch
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

from valetts.data.preprocessing.audio import AudioPreprocessor
from valetts.data.preprocessing.text_en import EnglishTextPreprocessor

logger = logging.getLogger(__name__)


class EnglishTTSDataset(Dataset):
    """
    Dataset TTS específico para inglês.
    
    Carrega e processa dados TTS com foco em inglês:
    - Usa EnglishTextPreprocessor para G2P correto
    - Filtra por locale='en' automaticamente
    - Suporte a phonemes específicos do inglês
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        metadata_file: Union[str, Path],
        audio_config: Dict[str, Any],
        text_config: Dict[str, Any],
        split: str = "train",
        max_samples: Optional[int] = None,
        min_duration: float = 0.5,
        max_duration: float = 15.0,
        expected_locale: str = "en",
        locale_column: str = "locale",
        audio_column: str = "audio_path",
        text_column: str = "text_normalized",
        speaker_column: str = "speaker_id",
    ):
        """
        Inicializa dataset TTS para inglês.
        
        Args:
            data_dir: Diretório base dos dados
            metadata_file: Arquivo CSV com metadados
            audio_config: Configuração do processador de áudio
            text_config: Configuração do processador de texto
            split: Split do dataset (train/val)
            max_samples: Máximo de amostras (para debug)
            min_duration: Duração mínima em segundos
            max_duration: Duração máxima em segundos
            expected_locale: Locale esperado no dataset
            locale_column: Nome da coluna de locale
            audio_column: Nome da coluna de caminho do áudio
            text_column: Nome da coluna de texto
            speaker_column: Nome da coluna de speaker
        """
        self.data_dir = Path(data_dir)
        self.metadata_file = Path(metadata_file)
        self.split = split
        self.max_samples = max_samples
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.expected_locale = expected_locale
        
        # Mapeamento de colunas
        self.locale_column = locale_column
        self.audio_column = audio_column
        self.text_column = text_column
        self.speaker_column = speaker_column
        
        # Inicializar processadores
        self.audio_processor = AudioPreprocessor(**audio_config)
        self.text_processor = EnglishTextPreprocessor(**text_config)
        
        # Carregar e filtrar dados
        self.samples = self._load_and_filter_data()
        
        # Construir mapeamento de speakers
        self._build_speaker_mapping()
        
        logger.info(f"✅ EnglishTTSDataset carregado:")
        logger.info(f"   📊 Split: {self.split}")
        logger.info(f"   📈 Amostras: {len(self.samples):,}")
        logger.info(f"   🎤 Speakers: {self.n_speakers}")
        logger.info(f"   🌍 Locale: {self.expected_locale}")

    def _load_and_filter_data(self) -> List[Dict[str, Any]]:
        """Carrega e filtra dados do CSV."""
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"❌ Arquivo não encontrado: {self.metadata_file}")
        
        # Carregar CSV
        df = pd.read_csv(self.metadata_file)
        logger.info(f"📋 CSV carregado: {len(df):,} amostras totais")
        
        # Verificar colunas obrigatórias
        required_cols = [self.audio_column, self.text_column, self.speaker_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ Colunas ausentes: {missing_cols}")
        
        # Filtrar por locale se coluna existe
        if self.locale_column in df.columns:
            df_en = df[df[self.locale_column] == self.expected_locale].copy()
            logger.info(f"🌍 Filtrado por locale='{self.expected_locale}': {len(df_en):,} amostras")
            if len(df_en) == 0:
                logger.warning(f"⚠️ Nenhuma amostra encontrada para locale='{self.expected_locale}'")
                logger.info(f"   Locales disponíveis: {df[self.locale_column].unique().tolist()}")
        else:
            df_en = df.copy()
            logger.warning(f"⚠️ Coluna '{self.locale_column}' não encontrada, usando todas as amostras")
        
        # Filtrar por duração se coluna existe
        if 'duration' in df_en.columns:
            initial_count = len(df_en)
            df_en = df_en[
                (df_en['duration'] >= self.min_duration) & 
                (df_en['duration'] <= self.max_duration)
            ].copy()
            logger.info(f"⏱️ Filtrado por duração ({self.min_duration}-{self.max_duration}s): {len(df_en):,}/{initial_count}")
        
        # Split automático se não há coluna split
        if 'split' in df_en.columns:
            df_split = df_en[df_en['split'] == self.split].copy()
            if len(df_split) == 0:
                logger.warning(f"⚠️ Split '{self.split}' vazio, fazendo split automático")
                df_split = self._auto_split(df_en)
        else:
            df_split = self._auto_split(df_en)
        
        logger.info(f"📊 Split '{self.split}': {len(df_split):,} amostras")
        
        # Limitar amostras para debug
        if self.max_samples and len(df_split) > self.max_samples:
            df_split = df_split.head(self.max_samples).copy()
            logger.info(f"🐛 Debug: limitado a {self.max_samples} amostras")
        
        # Validar caminhos de áudio
        valid_samples = []
        for _, row in df_split.iterrows():
            audio_path = self.data_dir / row[self.audio_column]
            if audio_path.exists():
                sample = {
                    'audio_path': audio_path,
                    'text': row[self.text_column],
                    'speaker_id': row[self.speaker_column],
                    'original_data': dict(row)
                }
                valid_samples.append(sample)
            else:
                logger.debug(f"⚠️ Áudio não encontrado: {audio_path}")
        
        logger.info(f"✅ Amostras válidas: {len(valid_samples):,}")
        return valid_samples

    def _auto_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """Faz split automático dos dados."""
        # Shuffle determinístico
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        total = len(df_shuffled)
        if self.split == "train":
            # 90% para treino
            end_idx = int(0.9 * total)
            return df_shuffled[:end_idx].copy()
        elif self.split == "val":
            # 10% para validação
            start_idx = int(0.9 * total)
            return df_shuffled[start_idx:].copy()
        else:
            # Split desconhecido, retornar tudo
            logger.warning(f"⚠️ Split '{self.split}' desconhecido, usando todos os dados")
            return df_shuffled.copy()

    def _build_speaker_mapping(self):
        """Constrói mapeamento de speakers."""
        speakers = sorted(set(sample['speaker_id'] for sample in self.samples))
        self.speaker_to_id = {speaker: i for i, speaker in enumerate(speakers)}
        self.id_to_speaker = {i: speaker for speaker, i in self.speaker_to_id.items()}
        self.n_speakers = len(speakers)
        
        logger.info(f"🎭 Speakers mapeados: {self.n_speakers}")
        if self.n_speakers <= 10:
            logger.info(f"   Lista: {speakers}")

    def __len__(self) -> int:
        """Retorna número de amostras."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retorna uma amostra processada.
        
        Returns:
            Dict com keys:
                - text: tensor com IDs dos tokens de texto
                - audio: tensor com mel-spectrogram
                - speaker_id: tensor com ID do speaker
                - text_length: comprimento do texto
                - audio_length: comprimento do áudio
        """
        sample = self.samples[idx]
        
        try:
            # Processar texto
            text = sample['text']
            
            # Usar EnglishTextPreprocessor
            text_ids = self.text_processor.encode(text, add_special_tokens=True)
            text_tensor = torch.LongTensor(text_ids)
            
            # Carregar e processar áudio
            audio_path = sample['audio_path']
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # Garantir mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)
            
            # Resample se necessário
            if sample_rate != self.audio_processor.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, self.audio_processor.sample_rate
                )
                waveform = resampler(waveform)
            
            # Gerar mel-spectrogram
            mel_spec = self.audio_processor.mel_spectrogram(waveform)
            
            # Speaker ID
            speaker_id = self.speaker_to_id[sample['speaker_id']]
            speaker_tensor = torch.LongTensor([speaker_id])
            
            return {
                'text': text_tensor,
                'audio': mel_spec,
                'speaker_id': speaker_tensor,
                'text_length': torch.LongTensor([len(text_ids)]),
                'audio_length': torch.LongTensor([mel_spec.shape[-1]]),
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar amostra {idx}: {e}")
            logger.error(f"   Arquivo: {sample['audio_path']}")
            logger.error(f"   Texto: {sample['text']}")
            raise


def create_english_dataloader(
    data_dir: Union[str, Path],
    metadata_file: Union[str, Path],
    audio_config: Dict[str, Any],
    text_config: Dict[str, Any],
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: Optional[bool] = None,
    **dataset_kwargs
) -> DataLoader:
    """
    Cria DataLoader para dataset TTS em inglês.
    
    Args:
        data_dir: Diretório dos dados
        metadata_file: Arquivo CSV com metadados
        audio_config: Configuração de áudio
        text_config: Configuração de texto
        split: Split do dataset
        batch_size: Tamanho do batch
        num_workers: Número de workers
        pin_memory: Usar pin_memory
        shuffle: Shuffle automático baseado no split
        **dataset_kwargs: Argumentos adicionais para o dataset
    
    Returns:
        DataLoader configurado
    """
    # Criar dataset
    dataset = EnglishTTSDataset(
        data_dir=data_dir,
        metadata_file=metadata_file,
        audio_config=audio_config,
        text_config=text_config,
        split=split,
        **dataset_kwargs
    )
    
    # Shuffle automático
    if shuffle is None:
        shuffle = (split == "train")
    
    # Função de collate personalizada
    def collate_fn(batch):
        """Collate function para batches de tamanhos variados."""
        # Separar componentes
        texts = [item['text'] for item in batch]
        audios = [item['audio'] for item in batch]
        speaker_ids = torch.cat([item['speaker_id'] for item in batch])
        text_lengths = torch.cat([item['text_length'] for item in batch])
        audio_lengths = torch.cat([item['audio_length'] for item in batch])
        
        # Pad textos
        max_text_len = max(text.size(0) for text in texts)
        padded_texts = torch.zeros(len(texts), max_text_len, dtype=torch.long)
        for i, text in enumerate(texts):
            padded_texts[i, :text.size(0)] = text
        
        # Pad áudios
        max_audio_len = max(audio.size(-1) for audio in audios)
        mel_channels = audios[0].size(0)
        padded_audios = torch.zeros(len(audios), mel_channels, max_audio_len)
        for i, audio in enumerate(audios):
            padded_audios[i, :, :audio.size(-1)] = audio
        
        return {
            'text': padded_texts,
            'audio': padded_audios,
            'speaker_id': speaker_ids,
            'text_length': text_lengths,
            'audio_length': audio_lengths,
        }
    
    # Criar DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    logger.info(f"✅ DataLoader criado:")
    logger.info(f"   📦 Batch size: {batch_size}")
    logger.info(f"   👷 Workers: {num_workers}")
    logger.info(f"   🔀 Shuffle: {shuffle}")
    logger.info(f"   📊 Batches: {len(dataloader)}")
    
    return dataloader