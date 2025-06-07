"""
Gerador e Processador de Texto
==============================

Sistema para geração, normalização e processamento de texto para TTS.
"""

import logging
import re
import unicodedata
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TextGenerator:
    """
    Gerador e processador de texto para TTS.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o processador de texto.

        Args:
            config: Configuração do processador
        """
        self.config = config
        self.min_length = config.get("min_length", 5)
        self.max_length = config.get("max_length", 300)
        self.filter_duplicates = config.get("filter_duplicates", True)

        logger.info("📝 TextGenerator inicializado")

    def normalize_text(self, text: str) -> str:
        """
        Normaliza texto para TTS.

        Args:
            text: Texto a normalizar

        Returns:
            Texto normalizado
        """
        # Remover acentos e caracteres especiais
        text = unicodedata.normalize("NFD", text)
        text = "".join(char for char in text if unicodedata.category(char) != "Mn")

        # Conversões básicas
        text = text.strip()
        text = re.sub(r"\s+", " ", text)  # Múltiplos espaços

        # Normalizar pontuação
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        text = re.sub(r"[–—]", "-", text)

        # Remover caracteres problemáticos
        text = re.sub(r"[^\w\s\.\,\!\?\;\:\'\"\-\(\)]", "", text)

        return text.strip()

    def validate_text(self, text: str) -> bool:
        """
        Valida se o texto é adequado para TTS.

        Args:
            text: Texto a validar

        Returns:
            True se válido, False caso contrário
        """
        if not text or len(text.strip()) < self.min_length:
            return False

        if len(text) > self.max_length:
            return False

        # Verificar se há pelo menos algumas letras
        if not re.search(r"[a-zA-Z]", text):
            return False

        # Verificar se não é só pontuação
        letters_count = len(re.findall(r"[a-zA-Z]", text))
        if letters_count < 3:
            return False

        return True

    def filter_texts(self, texts: List[str]) -> List[str]:
        """
        Filtra lista de textos aplicando validações.

        Args:
            texts: Lista de textos

        Returns:
            Lista de textos filtrados
        """
        filtered = []
        seen = set() if self.filter_duplicates else None

        for text in texts:
            normalized = self.normalize_text(text)

            if not self.validate_text(normalized):
                continue

            if self.filter_duplicates:
                if normalized.lower() in seen:
                    continue
                seen.add(normalized.lower())

            filtered.append(normalized)

        logger.info(f"📝 Filtrados {len(filtered)}/{len(texts)} textos")
        return filtered
