#!/usr/bin/env python3
"""
Script de Geração de Dataset TTS
================================

Script principal para gerar datasets usando Edge-TTS com falantes MultilingualNeural.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent))

from valetts.data.generation import DEFAULT_CONFIG, DatasetBuilder

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Gerar dataset TTS usando Edge-TTS")

    parser.add_argument(
        "--text-file",
        type=str,
        default="english-text-clean.txt",
        help="Arquivo de texto para usar como fonte",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/generated/edge_tts_dataset",
        help="Diretório de saída do dataset",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Máximo de amostras a gerar (None = todas)",
    )

    parser.add_argument(
        "--num-speakers", type=int, default=4, help="Número de falantes a usar"
    )

    parser.add_argument(
        "--test-mode", action="store_true", help="Modo de teste (apenas 20 amostras)"
    )

    args = parser.parse_args()

    # Verificar se arquivo de texto existe
    if not Path(args.text_file).exists():
        logger.error(f"❌ Arquivo de texto não encontrado: {args.text_file}")
        return

    # Configuração do dataset
    config = DEFAULT_CONFIG.copy()
    config["output_dir"] = args.output_dir

    # Modo de teste
    if args.test_mode:
        args.max_samples = 20
        logger.info("🧪 Modo de teste ativado - 20 amostras apenas")

    # Inicializar construtor
    logger.info("🚀 Iniciando geração de dataset TTS...")
    builder = DatasetBuilder(config)

    # Criar configuração de falantes
    speakers_config = builder.create_speaker_config(args.num_speakers)

    logger.info("📋 Configuração de falantes:")
    for speaker_id, speaker_config in speakers_config.items():
        voice = speaker_config["voice"]
        gender = speaker_config["gender"]
        style = speaker_config["style"]
        logger.info(f"   {speaker_id}: {voice} ({gender}, {style})")

    try:
        # Gerar dataset
        print("\n" + "=" * 60)
        print("🚀 INICIANDO GERAÇÃO DO DATASET")
        print("=" * 60)

        report = await builder.generate_from_text_file(
            text_file=args.text_file,
            speakers_config=speakers_config,
            max_samples=args.max_samples,
        )

        # Mostrar relatório
        print("\n" + "=" * 60)
        print("📊 RELATÓRIO FINAL DE GERAÇÃO")
        print("=" * 60)

        print(f"📈 Total de amostras: {report['total_samples']}")
        print(f"✅ Amostras bem-sucedidas: {report['successful_samples']}")
        print(f"📊 Taxa de sucesso: {report['success_rate']:.1%}")
        print(f"💾 Dataset salvo em: {report['dataset_path']}")

        print("\n🎙️ RESUMO POR FALANTE:")
        print("-" * 60)
        for speaker_id, stats in report["speaker_stats"].items():
            success_rate = stats["success_rate"]
            total = stats["total"]
            successful = stats["successful"]
            voice = (
                stats["config"]["voice"]
                .replace("en-US-", "")
                .replace("MultilingualNeural", "ML")
            )
            gender = stats["config"]["gender"]
            style = stats["config"]["style"]

            gender_emoji = "👨" if gender == "male" else "👩"
            print(
                f"{gender_emoji} {speaker_id}: {successful}/{total} ({success_rate:.1%})"
            )
            print(f"   🎵 {voice} ({gender}, {style})")

        print("\n" + "=" * 60)
        print("🎉 DATASET GERADO COM SUCESSO!")
        print("=" * 60)

    except Exception as e:
        logger.error(f"❌ Erro na geração: {e}")
        return


if __name__ == "__main__":
    try:
        # Instalar dependências se necessário
        try:
            import edge_tts
        except ImportError:
            logger.error("❌ edge-tts não instalado. Execute: pip install edge-tts")
            sys.exit(1)

        # Executar
        asyncio.run(main())

    except KeyboardInterrupt:
        logger.info("🛑 Interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro crítico: {e}")
        sys.exit(1)
