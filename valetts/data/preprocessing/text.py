"""
Preprocessamento de texto para ValeTTS.

Contém funcionalidades para:
- Normalização de texto (números, abreviações)
- Phonetização (grapheme-to-phoneme)
- Tokenização e encoding
- Suporte multilíngue básico
"""

import re
import torch
import unicodedata
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import json


class TextPreprocessor:
    """
    Preprocessador de texto para modelos TTS.

    Normaliza texto, converte para phonemas e tokeniza
    para uso em modelos de síntese de fala.
    """

    def __init__(
        self,
        language: str = "pt-br",
        vocab_size: int = 512,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        sos_token: str = "<SOS>",
        eos_token: str = "<EOS>",
        use_phonemes: bool = False,
        normalize_numbers: bool = True,
        normalize_whitespace: bool = True,
        lowercase: bool = True,
    ):
        """
        Inicializa o preprocessador de texto.

        Args:
            language: Código do idioma (pt-br, en-us, etc.)
            vocab_size: Tamanho do vocabulário
            pad_token: Token de padding
            unk_token: Token para palavras desconhecidas
            sos_token: Token de início de sequência
            eos_token: Token de fim de sequência
            use_phonemes: Se deve usar phonemas em vez de caracteres
            normalize_numbers: Se deve normalizar números
            normalize_whitespace: Se deve normalizar espaços em branco
            lowercase: Se deve converter para minúsculas
        """
        self.language = language
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.use_phonemes = use_phonemes
        self.normalize_numbers = normalize_numbers
        self.normalize_whitespace = normalize_whitespace
        self.lowercase = lowercase

        # Tokens especiais
        self.special_tokens = [pad_token, unk_token, sos_token, eos_token]

        # Vocabulário (será construído dinamicamente)
        self.vocab = {}
        self.reverse_vocab = {}
        self._build_base_vocab()

        # Patterns para normalização
        self._setup_normalization_patterns()

    def _build_base_vocab(self):
        """Constrói vocabulário base com tokens especiais."""
        self.vocab = {token: i for i, token in enumerate(self.special_tokens)}
        self.reverse_vocab = {i: token for token, i in self.vocab.items()}

        if not self.use_phonemes:
            # Caracteres básicos para português
            basic_chars = "abcdefghijklmnopqrstuvwxyz"
            basic_chars += "áàâãéêíóôõúç"
            basic_chars += "0123456789"
            basic_chars += " .,!?;:-()\"'"

            for char in basic_chars:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
                    self.reverse_vocab[len(self.reverse_vocab)] = char
        else:
            # Phonemas básicos para português (IPA subset)
            phonemes = [
                "a", "e", "i", "o", "u", "ɐ", "ɛ", "ɔ", "ɨ",  # Vogais
                "p", "b", "t", "d", "k", "g",  # Plosivas
                "f", "v", "s", "z", "ʃ", "ʒ", "x",  # Fricativas
                "m", "n", "ɲ", "ŋ",  # Nasais
                "l", "ʎ", "ɾ", "r",  # Liquidas
                "j", "w",  # Semivogais
                " ",  # Espaço
            ]

            for phoneme in phonemes:
                if phoneme not in self.vocab:
                    self.vocab[phoneme] = len(self.vocab)
                    self.reverse_vocab[len(self.reverse_vocab)] = phoneme

    def _setup_normalization_patterns(self):
        """Setup dos patterns de normalização de texto."""
        # Números por extenso (básico para português)
        self.number_words = {
            '0': 'zero', '1': 'um', '2': 'dois', '3': 'três', '4': 'quatro',
            '5': 'cinco', '6': 'seis', '7': 'sete', '8': 'oito', '9': 'nove',
            '10': 'dez', '11': 'onze', '12': 'doze', '13': 'treze', '14': 'quatorze',
            '15': 'quinze', '16': 'dezesseis', '17': 'dezessete', '18': 'dezoito',
            '19': 'dezenove', '20': 'vinte',
        }

        # Abreviações comuns
        self.abbreviations = {
            'dr.': 'doutor',
            'dra.': 'doutora',
            'sr.': 'senhor',
            'sra.': 'senhora',
            'prof.': 'professor',
            'profa.': 'professora',
            'av.': 'avenida',
            'r.': 'rua',
            'etc.': 'etcetera',
            'ex.': 'exemplo',
        }

        # Patterns de regex
        self.patterns = {
            'multiple_spaces': re.compile(r'\s+'),
            'number': re.compile(r'\b\d+\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'punctuation_spaces': re.compile(r'\s*([,.!?;:])\s*'),
        }

    def normalize_text(self, text: str) -> str:
        """
        Normaliza texto para TTS.

        Args:
            text: Texto de entrada

        Returns:
            Texto normalizado
        """
        # Remover acentos de controle
        text = unicodedata.normalize('NFKC', text)

        # Converter para minúsculas se configurado
        if self.lowercase:
            text = text.lower()

        # Normalizar números
        if self.normalize_numbers:
            text = self._normalize_numbers(text)

        # Expandir abreviações
        text = self._expand_abbreviations(text)

        # Remover/substituir caracteres indesejados
        text = self._clean_text(text)

        # Normalizar espaços em branco
        if self.normalize_whitespace:
            text = self.patterns['multiple_spaces'].sub(' ', text)
            text = text.strip()

        # Normalizar pontuação
        text = self.patterns['punctuation_spaces'].sub(r' \1 ', text)
        text = self.patterns['multiple_spaces'].sub(' ', text)

        return text

    def _normalize_numbers(self, text: str) -> str:
        """Converte números para texto."""
        def replace_number(match):
            num = match.group()
            if num in self.number_words:
                return self.number_words[num]
            elif len(num) <= 2:
                # Números de 21-99
                try:
                    n = int(num)
                    if 21 <= n <= 99:
                        tens = n // 10
                        ones = n % 10
                        tens_word = {
                            2: 'vinte', 3: 'trinta', 4: 'quarenta', 5: 'cinquenta',
                            6: 'sessenta', 7: 'setenta', 8: 'oitenta', 9: 'noventa'
                        }[tens]
                        if ones == 0:
                            return tens_word
                        else:
                            return f"{tens_word} e {self.number_words[str(ones)]}"
                except ValueError:
                    pass
            return num  # Manter número se não conseguir converter

        return self.patterns['number'].sub(replace_number, text)

    def _expand_abbreviations(self, text: str) -> str:
        """Expande abreviações."""
        for abbrev, expansion in self.abbreviations.items():
            text = text.replace(abbrev, expansion)
        return text

    def _clean_text(self, text: str) -> str:
        """Remove caracteres indesejados."""
        # Remover URLs e emails
        text = self.patterns['url'].sub(' ', text)
        text = self.patterns['email'].sub(' ', text)

        # Manter apenas caracteres do vocabulário
        if not self.use_phonemes:
            allowed_chars = set(self.vocab.keys()) - set(self.special_tokens)
            text = ''.join(char if char in allowed_chars else ' ' for char in text)

        return text

    def text_to_phonemes(self, text: str) -> str:
        """
        Converte texto para phonemas (implementação básica).

        Args:
            text: Texto normalizado

        Returns:
            String de phonemas
        """
        if not self.use_phonemes:
            return text

        # Implementação básica de G2P para português
        # Em produção, usar uma biblioteca especializada como phonemizer
        phoneme_map = {
            'a': 'a', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u',
            'ã': 'ɐ̃', 'õ': 'õ', 'é': 'e', 'ê': 'e', 'í': 'i',
            'ó': 'o', 'ô': 'o', 'ú': 'u', 'ç': 's',
            'lh': 'ʎ', 'nh': 'ɲ', 'rr': 'r', 'ss': 's',
            'ch': 'ʃ', 'x': 'ʃ', 'z': 'z', 'j': 'ʒ', 'g': 'ʒ',
        }

        # Aplicar mapeamento básico
        result = text
        for graph, phoneme in phoneme_map.items():
            result = result.replace(graph, phoneme)

        return result

    def tokenize(self, text: str) -> List[str]:
        """
        Tokeniza texto em unidades atômicas.

        Args:
            text: Texto processado

        Returns:
            Lista de tokens
        """
        if self.use_phonemes:
            # Tokenizar phonemas (cada phonema é um token)
            tokens = []
            i = 0
            while i < len(text):
                # Tentar phonemas multi-caractere primeiro
                found = False
                for length in [2, 1]:  # Tentar 2 chars, depois 1
                    if i + length <= len(text):
                        candidate = text[i:i+length]
                        if candidate in self.vocab:
                            tokens.append(candidate)
                            i += length
                            found = True
                            break
                if not found:
                    tokens.append(self.unk_token)
                    i += 1
        else:
            # Tokenizar caracteres
            tokens = list(text)

        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Codifica texto para IDs numéricos.

        Args:
            text: Texto de entrada
            add_special_tokens: Se deve adicionar tokens especiais

        Returns:
            Lista de IDs
        """
        # Preprocessar texto
        normalized_text = self.normalize_text(text)

        # Converter para phonemas se configurado
        if self.use_phonemes:
            normalized_text = self.text_to_phonemes(normalized_text)

        # Tokenizar
        tokens = self.tokenize(normalized_text)

        # Adicionar tokens especiais
        if add_special_tokens:
            tokens = [self.sos_token] + tokens + [self.eos_token]

        # Converter para IDs
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab[self.unk_token])

        return ids

    def decode(self, ids: List[int], remove_special_tokens: bool = True) -> str:
        """
        Decodifica IDs para texto.

        Args:
            ids: Lista de IDs
            remove_special_tokens: Se deve remover tokens especiais

        Returns:
            Texto decodificado
        """
        tokens = []
        for id in ids:
            if id in self.reverse_vocab:
                token = self.reverse_vocab[id]
                if remove_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
            else:
                tokens.append(self.unk_token)

        return ''.join(tokens)

    def pad_sequence(
        self,
        sequences: List[List[int]],
        max_length: Optional[int] = None,
        padding_side: str = "right"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica padding a sequências de IDs.

        Args:
            sequences: Lista de sequências de IDs
            max_length: Comprimento máximo (None = usar max da batch)
            padding_side: "left" ou "right"

        Returns:
            padded_sequences: Tensor com padding [batch_size, max_length]
            lengths: Comprimentos originais [batch_size]
        """
        lengths = [len(seq) for seq in sequences]
        max_len = max_length or max(lengths)

        batch_size = len(sequences)
        padded = torch.full((batch_size, max_len), self.vocab[self.pad_token], dtype=torch.long)

        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), max_len)
            if padding_side == "right":
                padded[i, :seq_len] = torch.tensor(seq[:seq_len])
            else:  # left
                padded[i, max_len-seq_len:] = torch.tensor(seq[:seq_len])

        return padded, torch.tensor(lengths)

    def build_vocab_from_texts(self, texts: List[str]):
        """
        Constrói vocabulário a partir de lista de textos.

        Args:
            texts: Lista de textos para análise
        """
        # Começar com vocabulário base
        char_counts = {}

        for text in texts:
            normalized = self.normalize_text(text)
            if self.use_phonemes:
                normalized = self.text_to_phonemes(normalized)

            tokens = self.tokenize(normalized)
            for token in tokens:
                char_counts[token] = char_counts.get(token, 0) + 1

        # Ordenar por frequência
        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)

        # Adicionar até vocab_size
        current_size = len(self.vocab)
        for char, count in sorted_chars:
            if char not in self.vocab and current_size < self.vocab_size:
                self.vocab[char] = current_size
                self.reverse_vocab[current_size] = char
                current_size += 1

    def save_vocab(self, path: Union[str, Path]):
        """
        Salva vocabulário em arquivo JSON.

        Args:
            path: Caminho para salvar
        """
        vocab_data = {
            'vocab': self.vocab,
            'config': {
                'language': self.language,
                'vocab_size': self.vocab_size,
                'use_phonemes': self.use_phonemes,
                'special_tokens': self.special_tokens,
            }
        }

        with open(str(path), 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: Union[str, Path]):
        """
        Carrega vocabulário de arquivo JSON.

        Args:
            path: Caminho do arquivo
        """
        with open(str(path), 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        self.vocab = vocab_data['vocab']
        self.reverse_vocab = {int(v): k for k, v in self.vocab.items()}

        # Atualizar configuração se disponível
        if 'config' in vocab_data:
            config = vocab_data['config']
            self.language = config.get('language', self.language)
            self.vocab_size = config.get('vocab_size', self.vocab_size)
            self.use_phonemes = config.get('use_phonemes', self.use_phonemes)

    def get_vocab_size(self) -> int:
        """Retorna tamanho do vocabulário."""
        return len(self.vocab)

    def get_config(self) -> dict:
        """
        Retorna configuração do preprocessador.

        Returns:
            Dict com parâmetros de configuração
        """
        return {
            'language': self.language,
            'vocab_size': self.vocab_size,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'sos_token': self.sos_token,
            'eos_token': self.eos_token,
            'use_phonemes': self.use_phonemes,
            'normalize_numbers': self.normalize_numbers,
            'normalize_whitespace': self.normalize_whitespace,
            'lowercase': self.lowercase,
        }
