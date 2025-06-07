"""
Utilitários de processamento de texto para o ValeTTS.
"""

import re
import string
import unicodedata
from typing import Dict, List, Optional, Tuple


class TextProcessor:
    """Processador de texto para TTS."""

    def __init__(
        self,
        language: str = "pt-br",
        symbols: Optional[List[str]] = None,
        pad_token: str = "_",
        unk_token: str = "~",
        eos_token: str = ".",
    ):
        """
        Inicializa o processador de texto.

        Args:
            language: Idioma do texto
            symbols: Lista de símbolos válidos
            pad_token: Token de padding
            unk_token: Token para caracteres desconhecidos
            eos_token: Token de fim de sequência
        """
        self.language = language
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_token = eos_token

        if symbols is None:
            # Símbolos padrão para português brasileiro
            self.symbols = self._get_default_symbols()
        else:
            self.symbols = symbols

        # Criar mapeamentos símbolo <-> índice
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def _get_default_symbols(self) -> List[str]:
        """Retorna símbolos padrão para português brasileiro."""
        # Caracteres básicos
        letters = "abcdefghijklmnopqrstuvwxyz"
        letters += "áàâãäéèêëíìîïóòôõöúùûüçñ"

        # Números
        numbers = "0123456789"

        # Pontuação
        punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

        # Espaços e tokens especiais
        special = [self.pad_token, self.unk_token, " "]

        # Combinar todos
        all_symbols = list(letters + numbers + punctuation) + special

        return sorted(list(set(all_symbols)))

    def normalize_text(self, text: str) -> str:
        """
        Normaliza o texto.

        Args:
            text: Texto para normalizar

        Returns:
            Texto normalizado
        """
        # Converter para minúsculas
        text = text.lower()

        # Normalizar unicode
        text = unicodedata.normalize("NFKD", text)

        # Remover caracteres de controle
        text = "".join(
            char
            for char in text
            if not unicodedata.category(char).startswith("C")
        )

        # Limpar espaços extras
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def text_to_sequence(self, text: str) -> List[int]:
        """
        Converte texto para sequência de IDs.

        Args:
            text: Texto para converter

        Returns:
            Lista de IDs dos símbolos
        """
        normalized_text = self.normalize_text(text)

        sequence = []
        for char in normalized_text:
            if char in self.symbol_to_id:
                sequence.append(self.symbol_to_id[char])
            else:
                sequence.append(self.symbol_to_id[self.unk_token])

        # Adicionar token de fim de sequência
        if self.eos_token in self.symbol_to_id:
            sequence.append(self.symbol_to_id[self.eos_token])

        return sequence

    def sequence_to_text(self, sequence: List[int]) -> str:
        """
        Converte sequência de IDs para texto.

        Args:
            sequence: Lista de IDs dos símbolos

        Returns:
            Texto reconstruído
        """
        text = ""
        for symbol_id in sequence:
            if symbol_id in self.id_to_symbol:
                symbol = self.id_to_symbol[symbol_id]
                if symbol not in [self.pad_token, self.eos_token]:
                    text += symbol

        return text

    def pad_sequence(self, sequence: List[int], max_length: int) -> List[int]:
        """
        Adiciona padding a uma sequência.

        Args:
            sequence: Sequência para fazer padding
            max_length: Comprimento máximo desejado

        Returns:
            Sequência com padding
        """
        pad_id = self.symbol_to_id[self.pad_token]

        if len(sequence) >= max_length:
            return sequence[:max_length]

        padded = sequence + [pad_id] * (max_length - len(sequence))
        return padded

    def clean_text(self, text: str) -> str:
        """
        Limpa o texto removendo caracteres inválidos.

        Args:
            text: Texto para limpar

        Returns:
            Texto limpo
        """
        # Manter apenas caracteres válidos
        cleaned = ""
        for char in text:
            if char in self.symbol_to_id or char == " ":
                cleaned += char
            else:
                cleaned += self.unk_token

        return cleaned

    @property
    def vocab_size(self) -> int:
        """Retorna o tamanho do vocabulário."""
        return len(self.symbols)

    def get_symbol_stats(self, text: str) -> Dict[str, int]:
        """
        Retorna estatísticas dos símbolos no texto.

        Args:
            text: Texto para analisar

        Returns:
            Dicionário com contagem de cada símbolo
        """
        stats = {}
        for char in text:
            stats[char] = stats.get(char, 0) + 1
        return stats
