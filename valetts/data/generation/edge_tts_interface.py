"""
Interface com Edge-TTS para síntese de voz
==========================================

Sistema para interagir com Edge-TTS, incluindo:
- Falantes MultilingualNeural americanos
- Síntese em lote
- Controle de qualidade
- Processamento assíncrono
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm.asyncio import tqdm

try:
    import edge_tts
except ImportError:
    edge_tts = None
    logging.warning("edge-tts não instalado. Execute: pip install edge-tts")

logger = logging.getLogger(__name__)


class EdgeTTSInterface:
    """
    Interface principal para Edge-TTS com suporte a falantes MultilingualNeural.
    """

    # Falantes americanos MultilingualNeural
    US_MULTILINGUAL_VOICES = {
        "male": [
            "en-US-AndrewMultilingualNeural",
            "en-US-BrianMultilingualNeural",
            "en-US-ChristopherNeural",  # Backup se multilingual não disponível
            "en-US-EricNeural",
            "en-US-GuyNeural",
            "en-US-RogerNeural",
        ],
        "female": [
            "en-US-AvaMultilingualNeural",
            "en-US-EmmaMultilingualNeural",
            "en-US-JennyMultilingualNeural",
            "en-US-AriaNeural",  # Backup se multilingual não disponível
            "en-US-DavisNeural",
            "en-US-JaneNeural",
            "en-US-NancyNeural",
            "en-US-SaraNeural",
        ],
    }

    # Estilos e configurações de voz
    VOICE_STYLES = {
        "neutral": {"rate": "0%", "pitch": "0%", "volume": "100%"},
        "cheerful": {"rate": "+5%", "pitch": "+2%", "volume": "100%"},
        "calm": {"rate": "-5%", "pitch": "-1%", "volume": "90%"},
        "professional": {"rate": "0%", "pitch": "0%", "volume": "95%"},
        "expressive": {"rate": "+3%", "pitch": "+1%", "volume": "100%"},
    }

    def __init__(self, output_dir: str = "data/generated/edge_tts"):
        """
        Inicializa a interface Edge-TTS.

        Args:
            output_dir: Diretório para salvar arquivos de áudio
        """
        if edge_tts is None:
            raise ImportError(
                "edge-tts não está instalado. Execute: pip install edge-tts"
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Estatísticas
        self.stats = {
            "total_generated": 0,
            "successful": 0,
            "failed": 0,
            "by_voice": {},
            "by_gender": {"male": 0, "female": 0},
        }

        logger.info(f"🎙️ EdgeTTSInterface inicializada - Output: {self.output_dir}")

    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Obtém lista de vozes disponíveis do Edge-TTS."""
        try:
            voices = await edge_tts.list_voices()

            # Filtrar apenas vozes americanas
            us_voices = [
                voice for voice in voices if voice["Locale"].startswith("en-US")
            ]

            logger.info(f"✅ Encontradas {len(us_voices)} vozes americanas")
            return us_voices

        except Exception as e:
            logger.error(f"❌ Erro ao obter vozes: {e}")
            return []

    def get_voice_by_gender_and_style(self, gender: str, style: str = "neutral") -> str:
        """
        Seleciona voz baseada no gênero e estilo.

        Args:
            gender: "male" ou "female"
            style: Estilo da voz (neutral, cheerful, etc.)

        Returns:
            Nome da voz selecionada
        """
        if gender not in self.US_MULTILINGUAL_VOICES:
            gender = "female"  # Default

        voices = self.US_MULTILINGUAL_VOICES[gender]

        # Priorizar vozes MultilingualNeural
        multilingual_voices = [v for v in voices if "Multilingual" in v]
        if multilingual_voices:
            return multilingual_voices[0]

        # Fallback para vozes normais
        return voices[0] if voices else "en-US-JennyNeural"

    def prepare_text(self, text: str) -> str:
        """
        Prepara texto para síntese (apenas limpeza básica).

        Args:
            text: Texto a ser sintetizado

        Returns:
            Texto limpo para Edge-TTS
        """
        # Apenas limpeza básica - Edge-TTS faz o resto
        cleaned_text = text.strip()
        return cleaned_text

    async def synthesize_text(
        self, text: str, voice: str, output_path: str, style: str = "neutral"
    ) -> bool:
        """
        Sintetiza um texto usando Edge-TTS.

        Args:
            text: Texto para sintetizar
            voice: Nome da voz
            output_path: Caminho do arquivo de saída
            style: Estilo de fala

        Returns:
            True se bem-sucedido, False caso contrário
        """
        try:
            # Limpar texto e enviar para Edge-TTS
            cleaned_text = self.prepare_text(text)
            communicate = edge_tts.Communicate(cleaned_text, voice)

            # Gerar áudio
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            await communicate.save(str(output_file))

            # Atualizar estatísticas
            self._update_stats(voice, True)

            logger.debug(f"✅ Áudio gerado: {output_file}")
            return True

        except Exception as e:
            logger.error(f"❌ Erro na síntese: {e}")
            self._update_stats(voice, False)
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
            voice_config: Configuração de voz {"gender": "male/female", "style": "neutral"}
            output_dir: Diretório de saída
            filename_prefix: Prefixo dos arquivos

        Returns:
            Lista de tuplas (texto, arquivo_audio, sucesso)
        """
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Selecionar voz
        voice = self.get_voice_by_gender_and_style(
            voice_config.get("gender", "female"), voice_config.get("style", "neutral")
        )

        voice_short = voice.replace("en-US-", "").replace("MultilingualNeural", "ML")
        logger.info(f"🎙️ Síntese em lote - {len(texts)} textos com voz: {voice_short}")

        # Processar cada texto com barra de progresso
        pbar_desc = f"🎙️ {filename_prefix} ({voice_short})"

        for i, text in tqdm(
            enumerate(texts),
            total=len(texts),
            desc=pbar_desc,
            unit="texto",
            ncols=100,
            leave=False,
        ):
            filename = f"{filename_prefix}_{i+1:04d}.wav"
            audio_path = output_path / filename

            # Mostrar texto atual (truncado)
            text_preview = text[:50] + "..." if len(text) > 50 else text
            tqdm.write(f"   📝 {i+1:3d}/{len(texts)} → {text_preview}")

            success = await self.synthesize_text(
                text=text,
                voice=voice,
                output_path=str(audio_path),
                style=voice_config.get("style", "neutral"),
            )

            results.append((text, str(audio_path), success))

            if not success:
                tqdm.write(f"   ❌ Falha na síntese: {text_preview}")

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
                {
                    "speaker_001": {"gender": "male", "style": "neutral"},
                    "speaker_002": {"gender": "female", "style": "cheerful"}
                }
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

    def _update_stats(self, voice: str, success: bool):
        """Atualiza estatísticas internas."""
        self.stats["total_generated"] += 1

        if success:
            self.stats["successful"] += 1
        else:
            self.stats["failed"] += 1

        # Stats por voz
        if voice not in self.stats["by_voice"]:
            self.stats["by_voice"][voice] = {"success": 0, "failed": 0}

        if success:
            self.stats["by_voice"][voice]["success"] += 1
        else:
            self.stats["by_voice"][voice]["failed"] += 1

        # Stats por gênero
        for gender, voices in self.US_MULTILINGUAL_VOICES.items():
            if voice in voices:
                if success:
                    self.stats["by_gender"][gender] += 1
                break

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de uso."""
        return self.stats.copy()

    def reset_statistics(self):
        """Reseta estatísticas."""
        self.stats = {
            "total_generated": 0,
            "successful": 0,
            "failed": 0,
            "by_voice": {},
            "by_gender": {"male": 0, "female": 0},
        }


# Função utilitária para teste rápido
async def test_edge_tts():
    """Testa a interface Edge-TTS."""
    interface = EdgeTTSInterface()

    # Teste simples
    test_text = "Hello, this is a test of the Edge TTS interface."
    voice = interface.get_voice_by_gender_and_style("female", "neutral")

    success = await interface.synthesize_text(
        text=test_text, voice=voice, output_path="test_output.wav"
    )

    if success:
        print("✅ Teste bem-sucedido!")
    else:
        print("❌ Teste falhou!")

    # Mostrar estatísticas
    stats = interface.get_statistics()
    print(f"📊 Estatísticas: {stats}")


if __name__ == "__main__":
    # Executar teste
    asyncio.run(test_edge_tts())
