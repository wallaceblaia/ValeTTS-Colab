# Sistema de Geração de Amostras de Áudio
# Permite validação qualitativa durante o treinamento

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchaudio

logger = logging.getLogger(__name__)


class AudioSampleGenerator:
    """Gera amostras de áudio durante o treinamento para validação qualitativa."""

    def __init__(
        self,
        output_dir: str = "samples",
        sample_rate: int = 22050,
        max_length: int = 1000,  # máximo tokens de texto
        language: str = "pt-br",  # Idioma para os textos de teste
    ):
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.language = language.lower()

        # Criar diretório de saída
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Textos de teste baseados no idioma
        if self.language in ["en", "en-us", "en-gb"]:
            # Textos de teste em inglês
            self.test_texts = [
                "Hello, I am the ValeTTS text-to-speech synthesis system.",
                "This is a quality test of the text-to-speech model.",
                "The training is progressing very well today.",
                "Artificial intelligence and natural language processing.",
                "Voice synthesis technology is evolving rapidly.",
                "The weather is beautiful today in the countryside.",
                "Machine learning models require careful tuning.",
                "Speech synthesis enables computers to communicate naturally.",
            ]
        else:
            # Textos de teste padrão em português brasileiro
            self.test_texts = [
                "Olá, eu sou o sistema de síntese de voz ValeTTS.",
                "Este é um teste de qualidade do modelo de texto para fala.",
                "O treinamento está progredindo muito bem hoje.",
                "Inteligência artificial e processamento de linguagem natural.",
                "A tecnologia de síntese de voz está evoluindo rapidamente.",
            ]

        # Informações dos falantes - será atualizado dinamicamente
        self.speakers = {}
        self._default_speakers = {
            "speaker_001": "AndrewML",  # Masculino
            "speaker_002": "BrianML",  # Masculino
            "speaker_003": "AvaML",  # Feminino
            "speaker_004": "EmmaML",  # Feminino
        }

    def set_speakers_from_dataset(self, speaker_ids: List[str]) -> None:
        """Configura speakers baseado nos IDs do dataset real."""
        self.speakers = {}
        for i, speaker_id in enumerate(sorted(speaker_ids)):
            # Usar nomes mais genéricos para speakers do dataset
            self.speakers[speaker_id] = f"Speaker_{i+1:03d}"

        logger.info(
            f"🎤 Configurados {len(self.speakers)} speakers para geração de amostras"
        )

    def generate_samples(
        self,
        model,
        epoch: int,
        text_preprocessor,
        device: torch.device,
        num_samples_per_speaker: int = 1,
        max_speakers: int = 5,  # Limitar número de speakers para não sobrecarregar
    ) -> Dict[str, Any]:
        """
        Gera amostras de áudio para todos os falantes.

        Args:
            model: Modelo VITS2 treinado
            epoch: Época atual
            text_preprocessor: Preprocessador de texto
            device: Device (cuda/cpu)
            num_samples_per_speaker: Quantas amostras por falante
            max_speakers: Máximo de speakers para gerar amostras

        Returns:
            Dict com informações das amostras geradas
        """
        try:
            logger.info(f"🎵 Gerando amostras de áudio - Época {epoch}")

            # Se não há speakers configurados, usar padrão
            speakers_to_use = (
                self.speakers if self.speakers else self._default_speakers
            )

            # Limitar número de speakers
            speakers_list = list(speakers_to_use.items())[:max_speakers]

            # Criar diretório específico da época
            epoch_dir = self.output_dir / f"epoch_{epoch:03d}"
            epoch_dir.mkdir(exist_ok=True)

            model.eval()
            generated_samples = []

            with torch.no_grad():
                for speaker_id, speaker_name in speakers_list:
                    logger.info(
                        f"   🎤 Gerando para {speaker_name} ({speaker_id})"
                    )

                    for i in range(num_samples_per_speaker):
                        # Selecionar texto (rodar entre os textos disponíveis)
                        text_idx = (len(generated_samples) + i) % len(
                            self.test_texts
                        )
                        text = self.test_texts[text_idx]

                        try:
                            # Gerar amostra
                            audio_path, metadata = (
                                self._generate_single_sample(
                                    model=model,
                                    text=text,
                                    speaker_id=speaker_id,
                                    speaker_name=speaker_name,
                                    epoch=epoch,
                                    sample_idx=i,
                                    epoch_dir=epoch_dir,
                                    text_preprocessor=text_preprocessor,
                                    device=device,
                                )
                            )

                            if audio_path:
                                generated_samples.append(
                                    {
                                        "speaker_id": speaker_id,
                                        "speaker_name": speaker_name,
                                        "text": text,
                                        "audio_path": str(audio_path),
                                        "metadata": metadata,
                                    }
                                )

                        except Exception as e:
                            logger.error(
                                f"❌ Erro gerando amostra para {speaker_name}: {e}"
                            )
                            continue

            model.train()

            # Salvar informações das amostras
            summary = {
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(generated_samples),
                "language": self.language,
                "speakers_used": len(speakers_list),
                "samples": generated_samples,
                "output_directory": str(epoch_dir),
            }

            # Salvar resumo
            summary_path = epoch_dir / "samples_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logger.info(
                f"✅ Geradas {len(generated_samples)} amostras em {epoch_dir}"
            )

            return summary

        except Exception as e:
            logger.error(f"❌ Erro na geração de amostras: {e}")
            return {"epoch": epoch, "error": str(e), "total_samples": 0}

    def _generate_single_sample(
        self,
        model,
        text: str,
        speaker_id: str,
        speaker_name: str,
        epoch: int,
        sample_idx: int,
        epoch_dir: Path,
        text_preprocessor,
        device: torch.device,
    ) -> tuple[Optional[Path], Dict[str, Any]]:
        """Gera uma única amostra de áudio."""
        try:
            # Preprocessar texto
            text_normalized = text_preprocessor.normalize_text(text)
            text_tokens = text_preprocessor.encode(text_normalized)

            # Converter para tensor
            if isinstance(text_tokens, list):
                text_tensor = (
                    torch.LongTensor(text_tokens).unsqueeze(0).to(device)
                )
            else:
                text_tensor = (
                    text_tokens.unsqueeze(0).to(device)
                    if text_tokens.dim() == 1
                    else text_tokens.to(device)
                )

            # Obter speaker embedding (assumindo que speaker_ids são mapeados para índices)
            speaker_mapping = {
                "speaker_001": 0,
                "speaker_002": 1,
                "speaker_003": 2,
                "speaker_004": 3,
            }

            speaker_idx = speaker_mapping.get(speaker_id, 0)
            speaker_tensor = torch.LongTensor([speaker_idx]).to(device)

            # Gerar áudio
            start_time = (
                torch.cuda.Event(enable_timing=True)
                if device.type == "cuda"
                else None
            )
            end_time = (
                torch.cuda.Event(enable_timing=True)
                if device.type == "cuda"
                else None
            )

            if start_time:
                start_time.record()

            # Inferência do modelo
            with torch.amp.autocast("cuda", enabled=True):
                text_lengths = torch.LongTensor([text_tensor.size(1)]).to(
                    device
                )
                audio_output = model.synthesize(
                    text=text_tensor,
                    text_lengths=text_lengths,
                    speaker_ids=speaker_tensor,
                )

            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                inference_time = (
                    start_time.elapsed_time(end_time) / 1000.0
                )  # em segundos
            else:
                inference_time = 0.0

            # Processar saída do modelo
            if isinstance(audio_output, tuple):
                audio = audio_output[0]  # Pegar apenas o áudio
            else:
                audio = audio_output

            # Converter para CPU e numpy se necessário
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().squeeze()

            # Garantir que é 1D
            if audio.dim() > 1:
                audio = audio.squeeze()

            # Normalizar áudio
            if audio.abs().max() > 0:
                audio = audio / audio.abs().max() * 0.95

            # Converter para float32 para compatibilidade com torchaudio.save
            audio = audio.float()

            # Nome do arquivo
            filename = f"{speaker_name.lower()}_sample{sample_idx:02d}_epoch{epoch:03d}.wav"
            audio_path = epoch_dir / filename

            # Salvar áudio
            torchaudio.save(
                str(audio_path),
                audio.unsqueeze(0),  # Adicionar dimensão batch
                self.sample_rate,
            )

            # Metadados
            metadata = {
                "text": text,
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "epoch": epoch,
                "sample_idx": sample_idx,
                "audio_length_seconds": len(audio) / self.sample_rate,
                "inference_time_seconds": inference_time,
                "sample_rate": self.sample_rate,
                "text_length": len(text),
                "audio_samples": len(audio),
            }

            return audio_path, metadata

        except Exception as e:
            logger.error(f"❌ Erro gerando amostra individual: {e}")
            return None, {"error": str(e)}

    def cleanup_old_samples(self, keep_last_n_epochs: int = 5) -> None:
        """Remove amostras antigas para economizar espaço."""
        try:
            # Listar todos os diretórios de época
            epoch_dirs = [
                d
                for d in self.output_dir.iterdir()
                if d.is_dir() and d.name.startswith("epoch_")
            ]

            if len(epoch_dirs) <= keep_last_n_epochs:
                return

            # Ordenar por número da época
            epoch_dirs.sort(key=lambda x: int(x.name.split("_")[1]))

            # Remover os mais antigos
            dirs_to_remove = epoch_dirs[:-keep_last_n_epochs]

            for dir_to_remove in dirs_to_remove:
                logger.info(f"🗑️ Removendo amostras antigas: {dir_to_remove}")
                import shutil

                shutil.rmtree(dir_to_remove)

        except Exception as e:
            logger.error(f"❌ Erro limpando amostras antigas: {e}")

    def get_latest_samples_info(self) -> Optional[Dict[str, Any]]:
        """Retorna informações das amostras mais recentes."""
        try:
            # Encontrar diretório mais recente
            epoch_dirs = [
                d
                for d in self.output_dir.iterdir()
                if d.is_dir() and d.name.startswith("epoch_")
            ]

            if not epoch_dirs:
                return None

            # Ordenar e pegar o mais recente
            latest_dir = max(
                epoch_dirs, key=lambda x: int(x.name.split("_")[1])
            )

            # Carregar resumo
            summary_path = latest_dir / "samples_summary.json"
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    return json.load(f)

            return None

        except Exception as e:
            logger.error(f"❌ Erro obtendo informações das amostras: {e}")
            return None
