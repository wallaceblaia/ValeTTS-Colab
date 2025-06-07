"""
Preprocessamento de texto para inglês - ValeTTS.

Contém funcionalidades específicas para inglês:
- Normalização de texto (números, abreviações em inglês)
- Phonetização G2P para inglês
- Tokenização e encoding
- Suporte específico para inglês americano/britânico
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch


class EnglishTextPreprocessor:
    """
    Preprocessador de texto específico para inglês.

    Normaliza texto, converte para phonemas e tokeniza
    para uso em modelos TTS com datasets em inglês.
    """

    def __init__(
        self,
        language: str = "en-us",
        vocab_size: int = 512,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        sos_token: str = "<SOS>",
        eos_token: str = "<EOS>",
        use_phonemes: bool = True,
        normalize_numbers: bool = True,
        normalize_whitespace: bool = True,
        lowercase: bool = True,
    ):
        """
        Inicializa o preprocessador de texto para inglês.

        Args:
            language: Código do idioma (en-us, en-gb, etc.)
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
            # Caracteres básicos para inglês
            basic_chars = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?;:-()\"'"
            for char in basic_chars:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
                    self.reverse_vocab[len(self.reverse_vocab)] = char
        else:
            # Phonemas para inglês (ARPAbet + IPA)
            english_phonemes = [
                # Vogais ARPAbet
                "AA",
                "AE",
                "AH",
                "AO",
                "AW",
                "AY",
                "EH",
                "ER",
                "EY",
                "IH",
                "IY",
                "OW",
                "OY",
                "UH",
                "UW",
                # Consoantes ARPAbet
                "B",
                "CH",
                "D",
                "DH",
                "F",
                "G",
                "HH",
                "JH",
                "K",
                "L",
                "M",
                "N",
                "NG",
                "P",
                "R",
                "S",
                "SH",
                "T",
                "TH",
                "V",
                "W",
                "Y",
                "Z",
                "ZH",
                # IPA equivalentes
                "i",
                "ɪ",
                "e",
                "ɛ",
                "æ",
                "ə",
                "ʌ",
                "ɑ",
                "ɔ",
                "o",
                "ʊ",
                "u",
                "aɪ",
                "eɪ",
                "ɔɪ",
                "aʊ",
                "oʊ",
                "p",
                "b",
                "t",
                "d",
                "k",
                "g",
                "f",
                "v",
                "θ",
                "ð",
                "s",
                "z",
                "ʃ",
                "ʒ",
                "tʃ",
                "dʒ",
                "m",
                "n",
                "ŋ",
                "l",
                "r",
                "w",
                "j",
                "h",
                " ",  # Espaço
            ]

            for phoneme in english_phonemes:
                if phoneme not in self.vocab:
                    self.vocab[phoneme] = len(self.vocab)
                    self.reverse_vocab[len(self.reverse_vocab)] = phoneme

    def _setup_normalization_patterns(self):
        """Setup dos patterns de normalização de texto para inglês."""
        # Números por extenso em inglês
        self.number_words = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
            "10": "ten",
            "11": "eleven",
            "12": "twelve",
            "13": "thirteen",
            "14": "fourteen",
            "15": "fifteen",
            "16": "sixteen",
            "17": "seventeen",
            "18": "eighteen",
            "19": "nineteen",
            "20": "twenty",
            "30": "thirty",
            "40": "forty",
            "50": "fifty",
            "60": "sixty",
            "70": "seventy",
            "80": "eighty",
            "90": "ninety",
        }

        # Abreviações em inglês
        self.abbreviations = {
            "dr.": "doctor",
            "mr.": "mister",
            "mrs.": "misses",
            "ms.": "miss",
            "prof.": "professor",
            "ave.": "avenue",
            "st.": "street",
            "rd.": "road",
            "blvd.": "boulevard",
            "etc.": "etcetera",
            "vs.": "versus",
            "e.g.": "for example",
            "i.e.": "that is",
            "inc.": "incorporated",
            "ltd.": "limited",
            "co.": "company",
            "corp.": "corporation",
        }

        # Patterns regex
        self.patterns = {
            "multiple_spaces": re.compile(r"\s+"),
            "number": re.compile(r"\b\d+\b"),
            "email": re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            ),
            "url": re.compile(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
            ),
            "punctuation_spaces": re.compile(r"\s*([,.!?;:])\s*"),
            "ordinal": re.compile(r"\b(\d+)(st|nd|rd|th)\b"),
        }

    def normalize_text(self, text: str) -> str:
        """
        Normaliza texto para TTS em inglês.

        Args:
            text: Texto de entrada

        Returns:
            Texto normalizado
        """
        # Remover acentos de controle
        text = unicodedata.normalize("NFKC", text)

        # Converter para minúsculas se configurado
        if self.lowercase:
            text = text.lower()

        # Normalizar números ordinais (1st -> first)
        text = self._normalize_ordinals(text)

        # Normalizar números
        if self.normalize_numbers:
            text = self._normalize_numbers(text)

        # Expandir abreviações
        text = self._expand_abbreviations(text)

        # Remover/substituir caracteres indesejados
        text = self._clean_text(text)

        # Normalizar espaços em branco
        if self.normalize_whitespace:
            text = self.patterns["multiple_spaces"].sub(" ", text)
            text = text.strip()

        # Normalizar pontuação
        text = self.patterns["punctuation_spaces"].sub(r" \1 ", text)
        text = self.patterns["multiple_spaces"].sub(" ", text)

        return text

    def _normalize_ordinals(self, text: str) -> str:
        """Converte números ordinais para texto."""
        ordinal_words = {
            "1st": "first",
            "2nd": "second",
            "3rd": "third",
            "4th": "fourth",
            "5th": "fifth",
            "6th": "sixth",
            "7th": "seventh",
            "8th": "eighth",
            "9th": "ninth",
            "10th": "tenth",
            "11th": "eleventh",
            "12th": "twelfth",
            "13th": "thirteenth",
            "14th": "fourteenth",
            "15th": "fifteenth",
            "16th": "sixteenth",
            "17th": "seventeenth",
            "18th": "eighteenth",
            "19th": "nineteenth",
            "20th": "twentieth",
            "21st": "twenty first",
        }

        def replace_ordinal(match):
            ordinal = match.group()
            return ordinal_words.get(ordinal, ordinal)

        return self.patterns["ordinal"].sub(replace_ordinal, text)

    def _normalize_numbers(self, text: str) -> str:
        """Converte números para texto em inglês."""

        def replace_number(match):
            num = match.group()
            if num in self.number_words:
                return self.number_words[num]
            elif len(num) <= 2:
                try:
                    n = int(num)
                    if 21 <= n <= 99:
                        tens = n // 10
                        ones = n % 10
                        tens_word = {
                            2: "twenty",
                            3: "thirty",
                            4: "forty",
                            5: "fifty",
                            6: "sixty",
                            7: "seventy",
                            8: "eighty",
                            9: "ninety",
                        }[tens]
                        if ones == 0:
                            return tens_word
                        else:
                            return (
                                f"{tens_word} {self.number_words[str(ones)]}"
                            )
                except ValueError:
                    pass
            elif len(num) == 3:
                # Centenas (100-999)
                try:
                    n = int(num)
                    hundreds = n // 100
                    remainder = n % 100
                    result = f"{self.number_words[str(hundreds)]} hundred"
                    if remainder > 0:
                        if remainder < 21:
                            result += f" {self.number_words[str(remainder)]}"
                        else:
                            tens = remainder // 10
                            ones = remainder % 10
                            tens_word = {
                                2: "twenty",
                                3: "thirty",
                                4: "forty",
                                5: "fifty",
                                6: "sixty",
                                7: "seventy",
                                8: "eighty",
                                9: "ninety",
                            }[tens]
                            if ones == 0:
                                result += f" {tens_word}"
                            else:
                                result += f" {tens_word} {self.number_words[str(ones)]}"
                    return result
                except ValueError:
                    pass
            return num  # Manter número se não conseguir converter

        return self.patterns["number"].sub(replace_number, text)

    def _expand_abbreviations(self, text: str) -> str:
        """Expande abreviações em inglês."""
        for abbrev, expansion in self.abbreviations.items():
            text = text.replace(abbrev, expansion)
        return text

    def _clean_text(self, text: str) -> str:
        """Remove caracteres indesejados."""
        # Remover URLs e emails
        text = self.patterns["url"].sub(" ", text)
        text = self.patterns["email"].sub(" ", text)

        # Manter apenas caracteres do vocabulário
        if not self.use_phonemes:
            allowed_chars = set(self.vocab.keys()) - set(self.special_tokens)
            text = "".join(
                char if char in allowed_chars else " " for char in text
            )

        return text

    def text_to_phonemes(self, text: str) -> str:
        """
        Converte texto para phonemas (G2P para inglês).

        Args:
            text: Texto normalizado

        Returns:
            String de phonemes
        """
        if not self.use_phonemes:
            return text

        # Tentar usar phonemizer se disponível
        try:
            from phonemizer import phonemize

            return phonemize(
                text,
                language="en-us",
                backend="espeak",
                strip=True,
                preserve_punctuation=True,
                with_stress=False,
            )
        except ImportError:
            print(
                "Warning: phonemizer não instalado. Usando mapeamento básico."
            )
            return self._basic_english_g2p(text)

    def _basic_english_g2p(self, text: str) -> str:
        """Mapeamento básico G2P para inglês (fallback)."""
        # Mapeamento muito básico - em produção usar phonemizer
        basic_map = {
            "a": "æ",
            "e": "ɛ",
            "i": "ɪ",
            "o": "ɑ",
            "u": "ʌ",
            "th": "θ",
            "sh": "ʃ",
            "ch": "tʃ",
            "ng": "ŋ",
            "ph": "f",
            "gh": "f",
            "ck": "k",
        }

        result = text
        for graph, phoneme in basic_map.items():
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
            # Tokenizar phonemes (cada phonema é um token)
            tokens = []
            i = 0
            while i < len(text):
                # Tentar phonemas multi-caractere primeiro
                found = False
                for length in [3, 2, 1]:  # Tentar 3, 2, depois 1 char
                    if i + length <= len(text):
                        candidate = text[i : i + length]
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

    def decode(
        self, ids: List[int], remove_special_tokens: bool = True
    ) -> str:
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

        return "".join(tokens)

    def pad_sequence(
        self,
        sequences: List[List[int]],
        max_length: Optional[int] = None,
        padding_side: str = "right",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica padding às sequências.

        Args:
            sequences: Lista de sequências de IDs
            max_length: Comprimento máximo (None para usar o máximo das sequências)
            padding_side: Lado do padding ('left' ou 'right')

        Returns:
            Tuple com (sequências_padded, máscaras_atenção)
        """
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)

        padded_sequences = []
        attention_masks = []

        for seq in sequences:
            seq_len = len(seq)
            pad_length = max_length - seq_len

            if pad_length > 0:
                padding = [self.vocab[self.pad_token]] * pad_length
                if padding_side == "right":
                    padded_seq = seq + padding
                    attention_mask = [1] * seq_len + [0] * pad_length
                else:
                    padded_seq = padding + seq
                    attention_mask = [0] * pad_length + [1] * seq_len
            else:
                padded_seq = seq[:max_length]
                attention_mask = [1] * max_length

            padded_sequences.append(padded_seq)
            attention_masks.append(attention_mask)

        return (
            torch.tensor(padded_sequences, dtype=torch.long),
            torch.tensor(attention_masks, dtype=torch.long),
        )

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
        sorted_chars = sorted(
            char_counts.items(), key=lambda x: x[1], reverse=True
        )

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
            "vocab": self.vocab,
            "config": {
                "language": self.language,
                "vocab_size": self.vocab_size,
                "use_phonemes": self.use_phonemes,
                "special_tokens": self.special_tokens,
            },
        }

        with open(str(path), "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: Union[str, Path]):
        """
        Carrega vocabulário de arquivo JSON.

        Args:
            path: Caminho do arquivo
        """
        with open(str(path), "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.vocab = vocab_data["vocab"]
        self.reverse_vocab = {int(v): k for k, v in self.vocab.items()}

        # Atualizar configuração se disponível
        if "config" in vocab_data:
            config = vocab_data["config"]
            self.language = config.get("language", self.language)
            self.vocab_size = config.get("vocab_size", self.vocab_size)
            self.use_phonemes = config.get("use_phonemes", self.use_phonemes)

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
            "language": self.language,
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "sos_token": self.sos_token,
            "eos_token": self.eos_token,
            "use_phonemes": self.use_phonemes,
            "normalize_numbers": self.normalize_numbers,
            "normalize_whitespace": self.normalize_whitespace,
            "lowercase": self.lowercase,
        }
