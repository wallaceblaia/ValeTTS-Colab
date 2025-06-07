#!/usr/bin/env python3
"""
Gerador de Dataset ElevenLabs com Controle de Progresso
======================================================

- Sistema de IDs para controle de progresso
- Teste inicial com uma linha por falante
- Unificação com dataset Edge-TTS existente
- Recuperação de sessão em caso de interrupção
"""

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent))

from valetts.data.generation.elevenlabs_interface import ElevenLabsInterface


class ElevenLabsDatasetGenerator:
    """Gerador de dataset ElevenLabs com controle avançado."""

    def __init__(
        self,
        texts_file: str,
        output_dir: str = "data/generated/elevenlabs",
        progress_file: str = "elevenlabs_progress.json",
    ):

        self.texts_file = Path(texts_file)
        self.output_dir = Path(output_dir)
        self.progress_file = Path(progress_file)
        self.session_id = str(uuid.uuid4())[:8]

        # Interface ElevenLabs
        self.interface = ElevenLabsInterface(output_dir=str(self.output_dir))

        # Configuração de falantes americanos
        self.speakers_config = {
            # Masculinos
            "liam_american": {"gender": "male", "age": "young", "style": "natural"},
            "will_american": {
                "gender": "male",
                "age": "young",
                "style": "conversational",
            },
            "eric_american": {
                "gender": "male",
                "age": "middle_aged",
                "style": "professional",
            },
            "chris_american": {
                "gender": "male",
                "age": "middle_aged",
                "style": "natural",
            },
            "brian_american": {
                "gender": "male",
                "age": "middle_aged",
                "style": "conversational",
            },
            "bill_american": {"gender": "male", "age": "old", "style": "professional"},
            # Femininas
            "aria_american": {
                "gender": "female",
                "age": "middle_aged",
                "style": "professional",
            },
            "sarah_american": {"gender": "female", "age": "young", "style": "natural"},
            "laura_american": {
                "gender": "female",
                "age": "young",
                "style": "expressive",
            },
            "matilda_american": {
                "gender": "female",
                "age": "middle_aged",
                "style": "professional",
            },
            "jessica_american": {
                "gender": "female",
                "age": "young",
                "style": "conversational",
            },
            "brittney_american": {
                "gender": "female",
                "age": "middle_aged",
                "style": "dramatic",
            },
            # Neutra
            "river_american": {
                "gender": "neutral",
                "age": "middle_aged",
                "style": "natural",
            },
        }

        print(f"🎙️ ElevenLabs Dataset Generator - Sessão: {self.session_id}")
        print(f"📁 Textos: {self.texts_file}")
        print(f"📁 Output: {self.output_dir}")
        print(f"👥 Falantes: {len(self.speakers_config)}")

    def load_texts_with_ids(self) -> List[Dict[str, Any]]:
        """Carrega textos do arquivo e adiciona IDs únicos."""

        if not self.texts_file.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.texts_file}")

        print(f"📖 Carregando textos de {self.texts_file}...")

        with open(self.texts_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        texts_with_ids = []
        for line_num, text in enumerate(lines, 1):
            text = text.strip()
            if text:  # Ignorar linhas vazias
                text_id = f"EL{line_num:04d}"  # EL0001, EL0002, etc.
                texts_with_ids.append(
                    {
                        "id": text_id,
                        "line_number": line_num,
                        "text": text,
                        "length": len(text),
                        "word_count": len(text.split()),
                    }
                )

        print(f"✅ Carregados {len(texts_with_ids)} textos válidos")

        # Salvar índice de textos
        index_file = self.output_dir / "texts_index.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(texts_with_ids, f, indent=2, ensure_ascii=False)

        print(f"💾 Índice salvo em: {index_file}")

        return texts_with_ids

    def load_progress(self) -> Dict[str, Any]:
        """Carrega progresso de sessão anterior."""

        if not self.progress_file.exists():
            return {
                "session_id": self.session_id,
                "start_time": datetime.now().isoformat(),
                "completed_texts": [],
                "failed_texts": [],
                "speakers_completed": {},
                "last_text_id": None,
                "last_speaker": None,
                "total_cost": 0.0,
                "total_characters": 0,
                "status": "starting",
            }

        print(f"📂 Carregando progresso de sessão anterior...")

        with open(self.progress_file, "r", encoding="utf-8") as f:
            progress = json.load(f)

        print(f"🔄 Sessão anterior: {progress.get('session_id', 'unknown')}")
        print(f"📊 Textos concluídos: {len(progress.get('completed_texts', []))}")
        print(f"❌ Textos falhados: {len(progress.get('failed_texts', []))}")

        return progress

    def save_progress(self, progress: Dict[str, Any]):
        """Salva progresso atual."""

        progress["last_update"] = datetime.now().isoformat()

        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

    async def test_single_samples(self, texts: List[Dict[str, Any]]) -> bool:
        """Testa com uma amostra por falante para validar qualidade."""

        print(f"\n🧪 TESTE INICIAL - UMA AMOSTRA POR FALANTE")
        print("=" * 60)

        # Usar primeira linha válida do arquivo
        test_text_data = texts[0]
        test_text = test_text_data["text"]

        print(f"📝 Texto de teste: {test_text[:80]}...")
        print(f"🆔 ID: {test_text_data['id']}")
        print(f"📏 Caracteres: {test_text_data['length']}")

        test_dir = self.output_dir / "test_samples"
        test_dir.mkdir(parents=True, exist_ok=True)

        results = []

        # Testar cada falante
        for speaker_id, config in self.speakers_config.items():
            print(f"\n🎙️ Testando falante: {speaker_id}")

            # Selecionar voz (remover 'style' dos critérios)
            voice_criteria = {k: v for k, v in config.items() if k != "style"}
            voice_info = self.interface.get_voice_by_criteria(**voice_criteria)
            if not voice_info:
                print(f"   ❌ Nenhuma voz encontrada para {voice_criteria}")
                continue

            print(f"   🎤 Voz: {voice_info['name']}")

            # Gerar áudio
            output_path = test_dir / f"{speaker_id}_test_{test_text_data['id']}.mp3"

            success = await self.interface.synthesize_text(
                text=test_text,
                voice_id=voice_info["voice_id"],
                output_path=str(output_path),
                style=config.get("style", "natural"),
            )

            result = {
                "speaker_id": speaker_id,
                "voice_name": voice_info["name"],
                "voice_id": voice_info["voice_id"],
                "config": config,
                "success": success,
                "output_path": str(output_path) if success else None,
                "text_id": test_text_data["id"],
                "text": test_text,
            }

            results.append(result)

            if success:
                print(f"   ✅ Sucesso: {output_path}")
            else:
                print(f"   ❌ Falha na síntese")

        # Salvar resultados do teste
        test_results_file = test_dir / "test_results.json"
        with open(test_results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Estatísticas
        successful = sum(1 for r in results if r["success"])
        total = len(results)

        print(f"\n📊 RESULTADOS DO TESTE:")
        print(f"   ✅ Sucessos: {successful}/{total}")
        print(f"   📁 Diretório: {test_dir}")
        print(f"   📄 Detalhes: {test_results_file}")

        if successful > 0:
            stats = self.interface.get_statistics()
            print(f"   💰 Custo estimado: {stats['estimated_cost_usd']}")
            print(f"   📝 Caracteres: {stats['total_characters']}")

            print(f"\n🎉 Teste concluído! Ouça os arquivos para verificar qualidade.")
            return True
        else:
            print(f"\n❌ Todos os testes falharam!")
            return False

    async def generate_full_dataset(
        self,
        texts: List[Dict[str, Any]],
        start_from: Optional[str] = None,
        max_texts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Gera dataset completo com controle de progresso."""

        print(f"\n🎬 GERANDO DATASET COMPLETO")
        print("=" * 60)

        # Carregar progresso
        progress = self.load_progress()

        # Determinar ponto de início
        completed_ids = set(progress.get("completed_texts", []))

        if start_from:
            start_idx = next(
                (i for i, t in enumerate(texts) if t["id"] == start_from), 0
            )
            print(f"🔄 Continuando do ID: {start_from}")
        else:
            start_idx = len(completed_ids)
            print(f"🔄 Continuando de onde parou: {start_idx} textos já concluídos")

        # Limitar quantidade se especificado
        if max_texts:
            end_idx = min(start_idx + max_texts, len(texts))
            texts_to_process = texts[start_idx:end_idx]
            print(
                f"📏 Processando {len(texts_to_process)} textos (limite: {max_texts})"
            )
        else:
            texts_to_process = texts[start_idx:]
            print(f"📏 Processando {len(texts_to_process)} textos restantes")

        if not texts_to_process:
            print(f"✅ Todos os textos já foram processados!")
            return progress

        total_cost_estimate = 0.0

        # Distribuir textos entre falantes (round-robin)
        speakers_list = list(self.speakers_config.keys())

        # Processar cada texto com APENAS UM falante (distribuição round-robin)
        for text_idx, text_data in enumerate(texts_to_process):
            text_id = text_data["id"]
            text_content = text_data["text"]

            if text_id in completed_ids:
                print(f"⏭️  Pulando texto já processado: {text_id}")
                continue

            # Determinar qual falante deve processar este texto
            # Usar linha original para consistência (não text_idx que pode mudar com start_from)
            line_number = text_data["line_number"]
            assigned_speaker_idx = (line_number - 1) % len(speakers_list)
            assigned_speaker = speakers_list[assigned_speaker_idx]
            config = self.speakers_config[assigned_speaker]

            print(f"\n📝 Processando texto {text_idx+1}/{len(texts_to_process)}")
            print(f"🆔 ID: {text_id} (linha {line_number})")
            print(f"🎙️ Falante atribuído: {assigned_speaker}")
            print(f"📄 Texto: {text_content[:100]}...")

            # Verificar se já foi processado
            speaker_key = f"{text_id}_{assigned_speaker}"
            if speaker_key in progress.get("speakers_completed", {}):
                print(f"   ⏭️  Já processado anteriormente")
                continue

            # Selecionar voz (remover 'style' dos critérios)
            voice_criteria = {k: v for k, v in config.items() if k != "style"}
            voice_info = self.interface.get_voice_by_criteria(**voice_criteria)
            if not voice_info:
                print(f"   ❌ Voz não encontrada para {voice_criteria}")
                progress.setdefault("failed_texts", []).append(text_id)
                continue

            print(f"   🎤 Voz: {voice_info['name']}")

            # Definir caminho de saída
            speaker_dir = self.output_dir / assigned_speaker / "raw"
            speaker_dir.mkdir(parents=True, exist_ok=True)

            output_path = speaker_dir / f"{assigned_speaker}_{text_id}.mp3"

            # Gerar áudio
            success = await self.interface.synthesize_text(
                text=text_content,
                voice_id=voice_info["voice_id"],
                output_path=str(output_path),
                style=config.get("style", "natural"),
            )

            if success:
                print(f"   ✅ Sucesso: {output_path.name}")

                # Registrar sucesso
                if "speakers_completed" not in progress:
                    progress["speakers_completed"] = {}
                progress["speakers_completed"][speaker_key] = {
                    "timestamp": datetime.now().isoformat(),
                    "output_path": str(output_path),
                    "voice_name": voice_info["name"],
                    "voice_id": voice_info["voice_id"],
                    "assigned_speaker": assigned_speaker,
                    "line_number": line_number,
                }

                # Marcar texto como concluído
                progress.setdefault("completed_texts", []).append(text_id)
                progress["last_text_id"] = text_id

                # Atualizar estatísticas
                stats = self.interface.get_statistics()
                progress["total_cost"] = stats.get("total_cost_estimate", 0.0)
                progress["total_characters"] = stats.get("total_characters", 0)

                print(f"   💰 Custo acumulado: ${progress['total_cost']:.4f}")
            else:
                print(f"   ❌ Falha na síntese")
                progress.setdefault("failed_texts", []).append(text_id)

            # Rate limiting
            await asyncio.sleep(0.6)  # 600ms entre requests

            # Salvar progresso a cada texto
            self.save_progress(progress)

        progress["status"] = "completed"
        progress["end_time"] = datetime.now().isoformat()
        self.save_progress(progress)

        return progress

    async def generate_metadata(self) -> Dict[str, Any]:
        """Gera metadados do dataset para unificação com Edge-TTS."""

        print(f"\n📋 GERANDO METADADOS DO DATASET")

        # Carregar progresso
        progress = self.load_progress()

        # Analisar arquivos gerados
        generated_files = []
        speakers_stats = {}

        for speaker_id in self.speakers_config.keys():
            speaker_dir = self.output_dir / speaker_id / "raw"

            if speaker_dir.exists():
                audio_files = list(speaker_dir.glob("*.mp3"))
                speakers_stats[speaker_id] = {
                    "total_files": len(audio_files),
                    "voice_config": self.speakers_config[speaker_id],
                    "files": [f.name for f in audio_files],
                }

                for audio_file in audio_files:
                    generated_files.append(
                        {
                            "speaker_id": speaker_id,
                            "file_path": str(audio_file),
                            "file_name": audio_file.name,
                            "text_id": audio_file.stem.split("_")[
                                -1
                            ],  # Extrair ID do texto
                        }
                    )

        metadata = {
            "dataset_info": {
                "name": "ValeTTS ElevenLabs Dataset",
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "source": "ElevenLabs API",
                "language": "en-US",
                "total_speakers": len(self.speakers_config),
                "total_files": len(generated_files),
            },
            "speakers": speakers_stats,
            "files": generated_files,
            "generation_stats": progress,
            "compatibility": {
                "format": "mp3",
                "sample_rate": "variable",
                "channels": "mono",
                "edge_tts_compatible": True,
            },
        }

        # Salvar metadados
        metadata_file = self.output_dir / "dataset_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✅ Metadados salvos em: {metadata_file}")
        print(f"📊 Total de arquivos: {len(generated_files)}")
        print(f"👥 Falantes: {len(speakers_stats)}")

        return metadata


async def main():
    """Função principal."""

    parser = argparse.ArgumentParser(description="Gerador de Dataset ElevenLabs")
    parser.add_argument(
        "--texts-file",
        type=str,
        default="scripts/elevenlabs.txt",
        help="Arquivo com textos para síntese",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/generated/elevenlabs",
        help="Diretório de saída",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Executar apenas teste inicial (uma linha por falante)",
    )
    parser.add_argument(
        "--start-from", type=str, help="ID do texto para continuar (ex: EL0100)"
    )
    parser.add_argument(
        "--max-texts", type=int, help="Máximo de textos para processar (para teste)"
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Gerar apenas metadados do dataset existente",
    )

    args = parser.parse_args()

    try:
        # Inicializar gerador
        generator = ElevenLabsDatasetGenerator(
            texts_file=args.texts_file, output_dir=args.output_dir
        )

        if args.metadata_only:
            # Gerar apenas metadados
            await generator.generate_metadata()
            return

        # Carregar textos
        texts = generator.load_texts_with_ids()

        if args.test_only:
            # Teste inicial apenas
            success = await generator.test_single_samples(texts)
            if success:
                print(
                    f"\n✅ Teste concluído! Verifique a qualidade dos áudios antes de continuar."
                )
                print(
                    f"🔊 Para continuar com dataset completo, execute sem --test-only"
                )
            return

        # Gerar dataset completo
        progress = await generator.generate_full_dataset(
            texts=texts, start_from=args.start_from, max_texts=args.max_texts
        )

        # Gerar metadados
        await generator.generate_metadata()

        # Resumo final
        print(f"\n🎉 GERAÇÃO CONCLUÍDA!")
        print(f"📊 Textos processados: {len(progress.get('completed_texts', []))}")
        print(f"❌ Textos falhados: {len(progress.get('failed_texts', []))}")
        print(f"💰 Custo total: ${progress.get('total_cost', 0):.4f}")
        print(f"📁 Dataset: {args.output_dir}")

    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
