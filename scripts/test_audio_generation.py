#!/usr/bin/env python3
"""
Script para testar geração de amostras de áudio usando checkpoints existentes.
"""

import argparse
import sys
from pathlib import Path

import torch

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent))

import pytorch_lightning as L

from valetts.data.preprocessing.text import TextPreprocessor
from valetts.models.vits2 import VITS2
from valetts.models.vits2.config import VITS2Config
from valetts.training.audio_sampling import AudioSampleGenerator
from valetts.training.trainers.vits2 import VITS2Trainer


def test_audio_generation(checkpoint_path: str, config_path: str):
    """Testa geração de áudio usando um checkpoint."""

    print(f"🎵 Testando geração de amostras de áudio")
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"📁 Config: {config_path}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    # Carregar configuração
    print("📋 Carregando configuração...")
    import yaml

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    # Extrair apenas a configuração do modelo
    model_config = full_config.get("model", {})
    data_config = full_config.get("data", {})

    # Mapear parâmetros para os nomes corretos do VITS2Config
    vits2_config = {
        "text_encoder_hidden_dim": model_config.get("text_encoder_hidden_dim", 192),
        "latent_dim": model_config.get("latent_dim", 192),
        "mel_channels": data_config.get("n_mels", 80),
        "n_speakers": model_config.get("n_speakers", 4),
        "speaker_embedding_dim": model_config.get("speaker_embedding_dim", 512),
        "generator_initial_channels": model_config.get(
            "generator_initial_channels", 512
        ),
        "sample_rate": data_config.get("sample_rate", 22050),
        "hop_length": data_config.get("hop_length", 256),
        "n_fft": data_config.get("n_fft", 1024),
        "win_length": data_config.get("win_length", 1024),
        "inference_only": model_config.get("inference_only", False),
    }

    config = VITS2Config(**vits2_config)

    # Criar modelo
    print("🤖 Criando modelo...")
    model = VITS2(config)

    # Carregar checkpoint
    print("💾 Carregando checkpoint...")
    trainer = VITS2Trainer.load_from_checkpoint(
        checkpoint_path, model=model, config=config.to_dict(), map_location=device
    )

    # Mover modelo para device e modo eval
    trainer.model.to(device)
    trainer.model.eval()

    # Criar preprocessador de texto
    print("📝 Criando preprocessador de texto...")
    text_preprocessor = TextPreprocessor(language="pt-br", vocab_size=512)

    # Criar gerador de amostras
    print("🎤 Criando gerador de amostras...")
    audio_sampler = AudioSampleGenerator(
        output_dir="test_samples", sample_rate=22050, max_length=1000
    )

    # Textos de teste
    test_texts = [
        "Olá, eu sou o sistema de síntese de fala ValeTTS.",
        "Este é um teste de geração de áudio.",
        "Como você está hoje?",
        "O tempo está muito bom para programar!",
        "Inteligência artificial é fascinante.",
    ]

    print(f"🎯 Gerando amostras para {len(test_texts)} textos...")

    # Gerar amostras
    try:
        summary = audio_sampler.generate_samples(
            model=trainer.model,
            epoch=999,  # Época de teste
            text_preprocessor=text_preprocessor,
            device=device,
            num_samples_per_speaker=1,
        )

        print(f"✅ Geração concluída!")
        print(f"📊 Total de amostras: {summary.get('total_samples', 0)}")
        print(f"📁 Diretório: {summary.get('output_directory', 'N/A')}")

        if summary.get("total_samples", 0) > 0:
            print(f"\n🎉 Sucesso! As amostras foram geradas corretamente.")
            print(f"🔊 Verifique os arquivos em: test_samples/epoch_999/")
        else:
            print(f"\n❌ Erro: Nenhuma amostra foi gerada.")

    except Exception as e:
        print(f"❌ Erro na geração: {e}")
        import traceback

        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Teste de geração de amostras de áudio"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/vits2_test/last.ckpt",
        help="Caminho para o checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/vits2_test_config.yaml",
        help="Caminho para a configuração",
    )

    args = parser.parse_args()

    # Verificar se arquivos existem
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint não encontrado: {args.checkpoint}")
        return

    if not Path(args.config).exists():
        print(f"❌ Config não encontrada: {args.config}")
        return

    test_audio_generation(args.checkpoint, args.config)


if __name__ == "__main__":
    main()
