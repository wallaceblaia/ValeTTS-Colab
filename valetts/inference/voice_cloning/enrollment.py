"""
Sistema de cadastro de speakers.
"""

import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


class SpeakerEnrollment:
    """Sistema de cadastro e gerenciamento de speakers."""

    def __init__(self):
        """Inicializa o sistema de enrollment."""
        self.speakers = {}
        logger.info("SpeakerEnrollment inicializado")

    def enroll(
        self,
        speaker_id: str,
        audio_samples: List[torch.Tensor],
        speaker_name: Optional[str] = None,
    ) -> None:
        """
        Cadastra um novo speaker.

        Args:
            speaker_id: ID único do speaker
            audio_samples: Lista de amostras de áudio
            speaker_name: Nome do speaker (opcional)
        """
        logger.info(f"Cadastrando speaker: {speaker_id}")

        # TODO: Extrair embeddings reais
        embedding = torch.randn(256)  # Embedding simulado

        self.speakers[speaker_id] = {
            "name": speaker_name or speaker_id,
            "samples": audio_samples,
            "embedding": embedding,
        }

        logger.info(f"Speaker {speaker_id} cadastrado com sucesso")

    def get_speaker_embedding(self, speaker_id: str) -> Optional[torch.Tensor]:
        """Retorna o embedding de um speaker."""
        if speaker_id in self.speakers:
            return self.speakers[speaker_id]["embedding"]
        return None
