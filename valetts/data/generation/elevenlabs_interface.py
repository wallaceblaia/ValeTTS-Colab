"""
Interface com ElevenLabs para síntese de voz
=============================================

Sistema para interagir com ElevenLabs API, incluindo:
- Vozes premium pré-definidas
- Síntese em lote
- Controle de qualidade
- Processamento assíncrono com suporte a português
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm.asyncio import tqdm

try:
    from elevenlabs import ElevenLabs, VoiceSettings
except ImportError:
    ElevenLabs = None
    VoiceSettings = None
    logging.warning("elevenlabs não instalado. Execute: uv add elevenlabs")

logger = logging.getLogger(__name__)


class ElevenLabsInterface:
    """
    Interface principal para ElevenLabs com suporte a vozes premium.
    """

    # Vozes americanas premium disponíveis organizadas por gênero
    AVAILABLE_VOICES = {
        "male": [
            {
                "id": "TX3LPaxmHKxFdv7VOQHJ",
                "name": "Liam",
                "accent": "american",
                "age": "young",
            },
            {
                "id": "bIHbv24MWmeRgasZH58o",
                "name": "Will",
                "accent": "american",
                "age": "young",
            },
            {
                "id": "cjVigY5qzO86Huf0OWal",
                "name": "Eric",
                "accent": "american",
                "age": "middle_aged",
            },
            {
                "id": "iP95p4xoKVk53GoZ742B",
                "name": "Chris",
                "accent": "american",
                "age": "middle_aged",
            },
            {
                "id": "nPczCjzI2devNBz1zQrb",
                "name": "Brian",
                "accent": "american",
                "age": "middle_aged",
            },
            {
                "id": "pqHfZKP75CvOlQylNhV4",
                "name": "Bill",
                "accent": "american",
                "age": "old",
            },
        ],
        "female": [
            {
                "id": "9BWtsMINqrJLrRacOk9x",
                "name": "Aria",
                "accent": "american",
                "age": "middle_aged",
            },
            {
                "id": "EXAVITQu4vr4xnSDxMaL",
                "name": "Sarah",
                "accent": "american",
                "age": "young",
            },
            {
                "id": "FGY2WhTYpPnrIDTdsKH5",
                "name": "Laura",
                "accent": "american",
                "age": "young",
            },
            {
                "id": "XrExE9yKIg1WjnnlVkGX",
                "name": "Matilda",
                "accent": "american",
                "age": "middle_aged",
            },
            {
                "id": "cgSgspJ2msm6clMCkdW9",
                "name": "Jessica",
                "accent": "american",
                "age": "young",
            },
            {
                "id": "esy0r39YPLQjOczyOib8",
                "name": "Brittney",
                "accent": "american",
                "age": "middle_aged",
            },
        ],
        "neutral": [
            {
                "id": "SAz9YHcvj6GT2YYXdXww",
                "name": "River",
                "accent": "american",
                "age": "middle_aged",
            },
        ],
    }

    # Configurações de qualidade por estilo
    VOICE_SETTINGS = {
        "natural": {
            "stability": 0.50,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
        },
        "expressive": {
            "stability": 0.30,
            "similarity_boost": 0.80,
            "style": 0.25,
            "use_speaker_boost": True,
        },
        "professional": {
            "stability": 0.70,
            "similarity_boost": 0.85,
            "style": 0.10,
            "use_speaker_boost": False,
        },
        "conversational": {
            "stability": 0.40,
            "similarity_boost": 0.70,
            "style": 0.15,
            "use_speaker_boost": True,
        },
        "dramatic": {
            "stability": 0.20,
            "similarity_boost": 0.90,
            "style": 0.40,
            "use_speaker_boost": True,
        },
    }

    def __init__(self, output_dir: str = "data/generated/elevenlabs"):
        """
        Inicializa a interface ElevenLabs.

        Args:
            output_dir: Diretório para salvar arquivos de áudio
        """
        if ElevenLabs is None:
            raise ImportError(
                "elevenlabs não está instalado. Execute: uv add elevenlabs"
            )

        # Verificar API key
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ELEVENLABS_API_KEY não encontrada no ambiente. "
                "Configure a variável de ambiente com sua chave da API."
            )

        self.client = ElevenLabs(api_key=self.api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Estatísticas
        self.stats = {
            "total_generated": 0,
            "successful": 0,
            "failed": 0,
            "by_voice": {},
            "by_gender": {"male": 0, "female": 0, "neutral": 0},
            "total_characters": 0,
            "total_cost_estimate": 0.0,
        }

        logger.info(f"🎙️ ElevenLabsInterface inicializada - Output: {self.output_dir}")

    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Obtém lista de vozes disponíveis organizadas."""
        all_voices = []

        for gender, voices in self.AVAILABLE_VOICES.items():
            for voice in voices:
                voice_info = {
                    "voice_id": voice["id"],
                    "name": voice["name"],
                    "gender": gender,
                    "accent": voice["accent"],
                    "age": voice["age"],
                    "category": "premade",
                }
                all_voices.append(voice_info)

        logger.info(f"✅ {len(all_voices)} vozes disponíveis carregadas")
        return all_voices

    def get_voice_by_criteria(
        self, gender: str = "female", accent: str = "american", age: str = "young"
    ) -> Optional[Dict[str, Any]]:
        """
        Seleciona voz baseada em critérios específicos.

        Args:
            gender: "male", "female" ou "neutral"
            accent: "american", "british", "australian", "swedish", "neutral"
            age: "young", "middle_aged", "old"

        Returns:
            Dicionário com dados da voz selecionada ou None
        """
        if gender not in self.AVAILABLE_VOICES:
            gender = "female"  # Default

        voices = self.AVAILABLE_VOICES[gender]

        # Filtrar por critérios
        candidates = []

        for voice in voices:
            if voice["accent"] == accent and voice["age"] == age:
                candidates.append(voice)

        # Se não encontrou com critérios exatos, relaxar filtros
        if not candidates:
            for voice in voices:
                if voice["accent"] == accent:
                    candidates.append(voice)

        if not candidates:
            candidates = voices  # Usar qualquer voz do gênero

        # Retornar primeira opção
        if candidates:
            selected = candidates[0]
            return {
                "voice_id": selected["id"],
                "name": selected["name"],
                "gender": gender,
                "accent": selected["accent"],
                "age": selected["age"],
            }

        return None

    def prepare_text(self, text: str) -> str:
        """
        Prepara texto para síntese com ElevenLabs.

        Args:
            text: Texto a ser sintetizado

        Returns:
            Texto otimizado para ElevenLabs
        """
        # Limpeza básica
        cleaned_text = text.strip()

        # ElevenLabs funciona bem com português sem modificações especiais
        return cleaned_text

    async def synthesize_text(
        self, text: str, voice_id: str, output_path: str, style: str = "natural"
    ) -> bool:
        """
        Sintetiza um texto usando ElevenLabs.

        Args:
            text: Texto para sintetizar
            voice_id: ID da voz
            output_path: Caminho do arquivo de saída
            style: Estilo de fala

        Returns:
            True se bem-sucedido, False caso contrário
        """
        try:
            # Preparar texto
            cleaned_text = self.prepare_text(text)

            # Configurações de voz
            voice_settings = self.VOICE_SETTINGS.get(
                style, self.VOICE_SETTINGS["natural"]
            )

            # Gerar áudio
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Síntese via ElevenLabs
            audio_generator = self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=cleaned_text,
                voice_settings=VoiceSettings(**voice_settings),
                model_id="eleven_multilingual_v2",  # Suporte melhor para português
            )

            # Salvar áudio
            with open(output_file, "wb") as f:
                for chunk in audio_generator:
                    f.write(chunk)

            # Atualizar estatísticas
            self._update_stats(voice_id, True, len(cleaned_text))

            logger.debug(f"✅ Áudio gerado: {output_file}")
            return True

        except Exception as e:
            logger.error(f"❌ Erro na síntese: {e}")
            self._update_stats(voice_id, False, len(text))
            return False

    async def synthesize_batch(
        self,
        texts: List[str],
        voice_config: Dict[str, Any],
        output_dir: str,
        filename_prefix: str = "sample",
    ) -> List[Tuple[str, str, bool]]:
        """
        Sintetiza múltiplos textos em lote.

        Args:
            texts: Lista de textos para sintetizar
            voice_config: Configuração de voz
            output_dir: Diretório de saída
            filename_prefix: Prefixo dos arquivos

        Returns:
            Lista de tuplas (texto, arquivo_audio, sucesso)
        """
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Selecionar voz
        voice_info = self.get_voice_by_criteria(
            gender=voice_config.get("gender", "female"),
            accent=voice_config.get("accent", "american"),
            age=voice_config.get("age", "young"),
        )

        if not voice_info:
            logger.error("❌ Nenhuma voz encontrada com os critérios especificados")
            return results

        voice_id = voice_info["voice_id"]
        voice_name = voice_info["name"]

        logger.info(f"🎙️ Síntese em lote - {len(texts)} textos com {voice_name}")

        # Processar cada texto com barra de progresso
        pbar_desc = f"🎙️ {filename_prefix} ({voice_name})"

        for i, text in tqdm(
            enumerate(texts),
            total=len(texts),
            desc=pbar_desc,
            unit="texto",
            ncols=100,
            leave=False,
        ):
            filename = f"{filename_prefix}_{i+1:04d}.mp3"  # ElevenLabs gera MP3
            audio_path = output_path / filename

            # Mostrar texto atual (truncado)
            text_preview = text[:50] + "..." if len(text) > 50 else text
            tqdm.write(f"   📝 {i+1:3d}/{len(texts)} → {text_preview}")

            success = await self.synthesize_text(
                text=text,
                voice_id=voice_id,
                output_path=str(audio_path),
                style=voice_config.get("style", "natural"),
            )

            results.append((text, str(audio_path), success))

            if not success:
                tqdm.write(f"   ❌ Falha na síntese: {text_preview}")

            # Rate limiting para evitar limites da API
            await asyncio.sleep(0.5)  # 500ms entre requests

        successful = sum(1 for _, _, success in results if success)
        logger.info(f"✅ Lote concluído - {successful}/{len(texts)} sucessos")

        return results

    async def generate_multi_speaker_dataset(
        self,
        texts: List[str],
        speakers_config: Dict[str, Dict[str, Any]],
        output_base_dir: str,
    ) -> Dict[str, List[Tuple[str, str, bool]]]:
        """
        Gera dataset com múltiplos falantes.

        Args:
            texts: Lista de textos
            speakers_config: Configuração de falantes
            output_base_dir: Diretório base de saída

        Returns:
            Dicionário com resultados por falante
        """
        all_results = {}
        base_path = Path(output_base_dir)

        logger.info(
            f"🎭 Gerando dataset multi-falante - {len(speakers_config)} falantes"
        )

        for speaker_id, config in speakers_config.items():
            logger.info(f"   🎙️ Processando falante: {speaker_id}")

            speaker_dir = base_path / speaker_id / "raw"

            results = await self.synthesize_batch(
                texts=texts,
                voice_config=config,
                output_dir=str(speaker_dir),
                filename_prefix=speaker_id,
            )

            all_results[speaker_id] = results

        logger.info(f"✅ Dataset multi-falante concluído")
        return all_results

    def _update_stats(self, voice_id: str, success: bool, char_count: int = 0):
        """Atualiza estatísticas internas."""
        self.stats["total_generated"] += 1
        self.stats["total_characters"] += char_count

        if success:
            self.stats["successful"] += 1
            # Estimativa de custo (ElevenLabs cobra por caractere)
            self.stats["total_cost_estimate"] += (
                char_count * 0.00003
            )  # ~$0.03 per 1K chars
        else:
            self.stats["failed"] += 1

        # Stats por voz
        if voice_id not in self.stats["by_voice"]:
            self.stats["by_voice"][voice_id] = {
                "success": 0,
                "failed": 0,
                "characters": 0,
            }

        if success:
            self.stats["by_voice"][voice_id]["success"] += 1
        else:
            self.stats["by_voice"][voice_id]["failed"] += 1

        self.stats["by_voice"][voice_id]["characters"] += char_count

        # Stats por gênero
        for gender, voices in self.AVAILABLE_VOICES.items():
            for voice in voices:
                if voice["id"] == voice_id:
                    if success:
                        self.stats["by_gender"][gender] += 1
                    break

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de uso."""
        stats = self.stats.copy()
        stats["estimated_cost_usd"] = f"${self.stats['total_cost_estimate']:.4f}"
        return stats

    def reset_statistics(self):
        """Reseta estatísticas."""
        self.stats = {
            "total_generated": 0,
            "successful": 0,
            "failed": 0,
            "by_voice": {},
            "by_gender": {"male": 0, "female": 0, "neutral": 0},
            "total_characters": 0,
            "total_cost_estimate": 0.0,
        }


# Função utilitária para teste rápido
async def test_elevenlabs():
    """Testa a interface ElevenLabs."""
    try:
        interface = ElevenLabsInterface()

        # Teste simples
        test_text = (
            "Olá, este é um teste da interface ElevenLabs em português brasileiro."
        )

        voice_info = interface.get_voice_by_criteria("female", "american", "young")
        if not voice_info:
            print("❌ Nenhuma voz disponível para teste")
            return

        success = await interface.synthesize_text(
            text=test_text,
            voice_id=voice_info["voice_id"],
            output_path="test_elevenlabs_output.mp3",
            style="natural",
        )

        if success:
            print(f"✅ Teste bem-sucedido com {voice_info['name']}!")
        else:
            print("❌ Teste falhou!")

        # Mostrar estatísticas
        stats = interface.get_statistics()
        print(f"📊 Estatísticas: {stats}")

    except Exception as e:
        print(f"❌ Erro no teste: {e}")


if __name__ == "__main__":
    # Executar teste
    asyncio.run(test_elevenlabs())
