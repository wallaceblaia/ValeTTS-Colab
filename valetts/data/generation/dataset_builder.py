"""
Construtor de Datasets TTS
=========================

Sistema principal para construção de datasets TTS usando Edge-TTS e outras fontes.
Integra geração de texto, síntese de áudio e criação de metadados.
"""

import asyncio
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from .audio_processor import AudioProcessor
from .edge_tts_interface import EdgeTTSInterface
from .text_generator import TextGenerator

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Construtor principal de datasets TTS.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o construtor de datasets.

        Args:
            config: Configuração do dataset
        """
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Componentes
        self.edge_tts = EdgeTTSInterface(
            output_dir=str(self.output_dir / "audio" / "raw")
        )
        self.text_gen = TextGenerator(config.get("text_config", {}))
        self.audio_proc = AudioProcessor(config.get("audio_config", {}))

        # Estatísticas
        self.stats = {
            "total_samples": 0,
            "successful_samples": 0,
            "failed_samples": 0,
            "speakers": {},
            "languages": {},
            "duration_stats": {"min": 0, "max": 0, "avg": 0, "total": 0},
        }

        logger.info(f"🏗️ DatasetBuilder inicializado - Output: {self.output_dir}")

    def create_speaker_config(self, num_speakers: int = 4) -> Dict[str, Dict[str, Any]]:
        """
        Cria configuração de falantes balanceada.

        Args:
            num_speakers: Número de falantes a criar

        Returns:
            Configuração de falantes
        """
        speakers_config = {}
        genders = ["male", "female"]
        styles = ["neutral", "cheerful", "calm", "professional"]

        for i in range(num_speakers):
            speaker_id = f"speaker_{i+1:03d}"
            gender = genders[i % 2]  # Alternar entre masculino e feminino
            style = styles[i % len(styles)]  # Rodar entre estilos

            speakers_config[speaker_id] = {
                "gender": gender,
                "style": style,
                "voice": self.edge_tts.get_voice_by_gender_and_style(gender, style),
            }

        logger.info(f"👥 Criada configuração para {num_speakers} falantes")
        return speakers_config

    async def generate_from_text_file(
        self,
        text_file: str,
        speakers_config: Optional[Dict[str, Dict[str, Any]]] = None,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Gera dataset a partir de arquivo de texto.

        Args:
            text_file: Caminho para arquivo de texto
            speakers_config: Configuração de falantes (auto-criada se None)
            max_samples: Limite máximo de amostras

        Returns:
            Relatório de geração
        """
        logger.info(f"📖 Carregando textos de: {text_file}")

        # Carregar textos
        texts = self._load_texts_from_file(text_file)

        if max_samples:
            texts = texts[:max_samples]

        logger.info(f"📝 {len(texts)} textos carregados")

        # Auto-criar configuração de falantes se não fornecida
        if speakers_config is None:
            num_speakers = min(4, max(2, len(texts) // 100))  # 1 speaker por 100 textos
            speakers_config = self.create_speaker_config(num_speakers)

        # Distribuir textos entre falantes
        distributed_texts = self._distribute_texts_among_speakers(
            texts, speakers_config
        )

        # Gerar áudio para cada falante
        all_results = {}
        total_speakers = len(distributed_texts)

        print(f"\n🎙️ Gerando dataset com {total_speakers} falantes...")

        for i, (speaker_id, speaker_data) in enumerate(distributed_texts.items(), 1):
            # Extrair apenas os textos das tuplas (linha, texto)
            speaker_texts = [text for line_num, text in speaker_data]

            # Info detalhada do falante
            speaker_config = speakers_config[speaker_id]
            voice_name = (
                speaker_config["voice"]
                .replace("en-US-", "")
                .replace("MultilingualNeural", "ML")
            )
            gender_emoji = "👨" if speaker_config["gender"] == "male" else "👩"

            print(f"\n{gender_emoji} FALANTE {i}/{total_speakers}: {speaker_id}")
            print(
                f"   🎵 Voz: {voice_name} ({speaker_config['gender']}, {speaker_config['style']})"
            )
            print(f"   📊 Textos: {len(speaker_texts)}")

            config = speakers_config[speaker_id]
            results = await self.edge_tts.synthesize_batch(
                texts=speaker_texts,
                voice_config=config,
                output_dir=str(self.output_dir / "audio" / "raw" / speaker_id),
                filename_prefix=speaker_id,
            )

            # Combinar resultados com números de linha originais
            results_with_lines = []
            for (line_num, original_text), (text, audio_path, success) in zip(
                speaker_data, results
            ):
                results_with_lines.append((line_num, text, audio_path, success))

            all_results[speaker_id] = results_with_lines

            # Stats do falante
            successful = sum(1 for _, _, _, success in results_with_lines if success)
            print(
                f"   ✅ Sucesso: {successful}/{len(speaker_texts)} ({successful/len(speaker_texts)*100:.1f}%)"
            )

        # Processar áudio
        await self._process_audio_files(all_results)

        # Criar metadados
        metadata = self._create_metadata(all_results, speakers_config)

        # Salvar metadados
        self._save_metadata(metadata)

        # Criar splits train/val/test
        self._create_data_splits(metadata)

        # Gerar relatório
        report = self._generate_report(all_results, speakers_config)

        logger.info("✅ Dataset gerado com sucesso!")
        return report

    def _load_texts_from_file(self, text_file: str) -> List[str]:
        """Carrega textos de arquivo."""
        texts = []
        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text and len(text) > 5:  # Filtrar textos muito curtos
                    texts.append(text)
        return texts

    def _distribute_texts_among_speakers(
        self, texts: List[str], speakers_config: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Distribui textos entre falantes mantendo ordem original."""
        distributed = {speaker_id: [] for speaker_id in speakers_config.keys()}
        speaker_ids = list(speakers_config.keys())

        # Distribuir em round-robin MANTENDO A ORDEM ORIGINAL
        # Linha 1 -> speaker_001, Linha 2 -> speaker_002, etc.
        for i, text in enumerate(texts):
            speaker_id = speaker_ids[i % len(speaker_ids)]
            distributed[speaker_id].append((i + 1, text))  # Incluir número da linha

        return distributed

    async def _process_audio_files(
        self, all_results: Dict[str, List[Tuple[int, str, str, bool]]]
    ):
        """Processa arquivos de áudio (normalização, conversão, etc.)."""
        logger.info("🔧 Processando arquivos de áudio...")

        for speaker_id, results in all_results.items():
            successful_files = [
                audio_path for _, _, audio_path, success in results if success
            ]

            if successful_files:
                # Processar com AudioProcessor
                processed_dir = self.output_dir / "audio" / "processed" / speaker_id
                await self.audio_proc.process_files_batch(
                    input_files=successful_files, output_dir=str(processed_dir)
                )

    def _create_metadata(
        self,
        all_results: Dict[str, List[Tuple[int, str, str, bool]]],
        speakers_config: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Cria metadados do dataset."""
        metadata = []

        for speaker_id, results in all_results.items():
            speaker_config = speakers_config[speaker_id]

            for i, (line_num, text, audio_path, success) in enumerate(results):
                if success:
                    # Usar número da linha original como ID único
                    sample_id = f"line_{line_num:04d}_{speaker_id}"

                    # Paths relativos
                    audio_rel_path = Path(audio_path).relative_to(self.output_dir)
                    processed_audio_path = (
                        self.output_dir
                        / "audio"
                        / "processed"
                        / speaker_id
                        / f"{speaker_id}_{i+1:04d}.wav"
                    )
                    processed_rel_path = processed_audio_path.relative_to(
                        self.output_dir
                    )

                    sample_meta = {
                        "id": sample_id,
                        "line_number": line_num,  # Número da linha original
                        "speaker_id": speaker_id,
                        "text": text,
                        "text_normalized": self.text_gen.normalize_text(text),
                        "audio_path": str(audio_rel_path),
                        "processed_audio_path": str(processed_rel_path),
                        "speaker_gender": speaker_config["gender"],
                        "speaker_style": speaker_config["style"],
                        "voice_name": speaker_config["voice"],
                        "language": "en-US",
                        "text_length": len(text),
                        "word_count": len(text.split()),
                        "duration": None,  # Será preenchido após processamento
                        "phonemes": None,  # Pode ser adicionado posteriormente
                        "created_at": datetime.now().isoformat(),
                    }

                    metadata.append(sample_meta)

        logger.info(f"📊 Metadados criados para {len(metadata)} amostras")
        return metadata

    def _save_metadata(self, metadata: List[Dict[str, Any]]):
        """Salva metadados em formato JSON e CSV."""
        # JSON completo
        json_path = self.output_dir / "metadata.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # CSV para análise
        df = pd.DataFrame(metadata)
        csv_path = self.output_dir / "metadata.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")

        # Salvar configuração
        config_path = self.output_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Metadados salvos: {json_path}, {csv_path}")

    def _create_data_splits(self, metadata: List[Dict[str, Any]]):
        """Cria splits train/validation/test."""
        # Configuração dos splits
        train_ratio = self.config.get("train_ratio", 0.8)
        val_ratio = self.config.get("val_ratio", 0.1)
        test_ratio = self.config.get("test_ratio", 0.1)

        # Embaralhar mantendo balanceamento por falante
        speakers = {}
        for item in metadata:
            speaker_id = item["speaker_id"]
            if speaker_id not in speakers:
                speakers[speaker_id] = []
            speakers[speaker_id].append(item)

        # Criar splits balanceados por falante
        train_data, val_data, test_data = [], [], []

        for speaker_id, speaker_samples in speakers.items():
            random.shuffle(speaker_samples)
            n_samples = len(speaker_samples)

            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)

            train_data.extend(speaker_samples[:n_train])
            val_data.extend(speaker_samples[n_train : n_train + n_val])
            test_data.extend(speaker_samples[n_train + n_val :])

        # Salvar splits
        splits_dir = self.output_dir / "splits"
        splits_dir.mkdir(exist_ok=True)

        for split_name, split_data in [
            ("train", train_data),
            ("val", val_data),
            ("test", test_data),
        ]:
            split_file = splits_dir / f"{split_name}.json"
            with open(split_file, "w", encoding="utf-8") as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"📂 Splits criados - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
        )

    def _generate_report(
        self,
        all_results: Dict[str, List[Tuple[int, str, str, bool]]],
        speakers_config: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Gera relatório de construção do dataset."""
        total_samples = sum(len(results) for results in all_results.values())
        successful_samples = sum(
            sum(1 for _, _, _, success in results if success)
            for results in all_results.values()
        )

        # Estatísticas por falante
        speaker_stats = {}
        for speaker_id, results in all_results.items():
            successful = sum(1 for _, _, _, success in results if success)
            speaker_stats[speaker_id] = {
                "total": len(results),
                "successful": successful,
                "success_rate": successful / len(results) if results else 0,
                "config": speakers_config[speaker_id],
            }

        # Estatísticas do Edge-TTS
        edge_stats = self.edge_tts.get_statistics()

        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset_path": str(self.output_dir),
            "config": self.config,
            "total_samples": total_samples,
            "successful_samples": successful_samples,
            "success_rate": successful_samples / total_samples if total_samples else 0,
            "speaker_stats": speaker_stats,
            "edge_tts_stats": edge_stats,
            "file_structure": {
                "audio_raw": str(self.output_dir / "audio" / "raw"),
                "audio_processed": str(self.output_dir / "audio" / "processed"),
                "metadata": str(self.output_dir / "metadata.json"),
                "splits": str(self.output_dir / "splits"),
            },
        }

        # Salvar relatório
        report_path = self.output_dir / "generation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"📋 Relatório salvo: {report_path}")
        return report


# Exemplo de configuração
DEFAULT_CONFIG = {
    "output_dir": "data/generated/dataset",
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "text_config": {"min_length": 5, "max_length": 300, "filter_duplicates": True},
    "audio_config": {"sample_rate": 22050, "normalize": True, "trim_silence": True},
}
