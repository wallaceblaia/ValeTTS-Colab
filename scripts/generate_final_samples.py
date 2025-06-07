#!/usr/bin/env python3
"""
Script para gerar amostras finais de áudio usando o checkpoint mais recente.
"""

import argparse
import sys
from pathlib import Path

import torch

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent))

from valetts.data.preprocessing.text import TextPreprocessor
from valetts.models.vits2 import VITS2
from valetts.models.vits2.config import VITS2Config
from valetts.training.audio_sampling import AudioSampleGenerator
from valetts.training.trainers.vits2 import VITS2Trainer


def generate_final_samples(
    checkpoint_path: str, config_path: str, output_dir: str = "final_samples"
):
    """Gera amostras finais com textos elaborados."""

    print(f"🎙️ Gerando amostras finais com checkpoint treinado")
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"📁 Config: {config_path}")
    print(f"📁 Output: {output_dir}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    # Carregar configuração
    print("📋 Carregando configuração...")
    import yaml

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    model_config = full_config.get("model", {})
    data_config = full_config.get("data", {})

    # Mapear parâmetros para VITS2Config
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

    # Textos elaborados para demonstração
    test_texts = [
        # Textos informativos
        "O sistema ValeTTS utiliza inteligência artificial para síntese de voz em português brasileiro.",
        "Este modelo foi treinado com arquitetura VITS2 e monitoramento automático por LLM.",
        # Textos técnicos
        "A rede neural convolucional processa espectrogramas mel com resolução temporal otimizada.",
        "O treinamento adversarial permite alta qualidade na geração de áudio sintético.",
        # Textos expressivos
        "Que maravilha! A tecnologia de síntese de voz está evoluindo rapidamente.",
        "Incrível como a inteligência artificial pode reproduzir a fala humana.",
        # Textos longos
        "O projeto ValeTTS representa um marco na síntese de voz em português brasileiro, combinando técnicas avançadas de aprendizado profundo com monitoramento inteligente em tempo real.",
        "A arquitetura VITS2 utiliza codificação variacional e treinamento adversarial para produzir áudio de alta qualidade com naturalidade impressionante.",
        # Textos com números e abreviações
        "O modelo possui 353 milhões de parâmetros e foi treinado por 10 épocas.",
        "Dr. Silva demonstrou que a taxa de amostragem de 22 mil e 50 hertz é adequada.",
        # Textos emotivos
        "Estou muito feliz com os resultados obtidos neste treinamento!",
        "Parabéns pela implementação deste sistema revolucionário de TTS.",
    ]

    # Criar gerador de amostras customizado
    print("🎤 Criando gerador de amostras...")
    audio_sampler = AudioSampleGenerator(
        output_dir=output_dir, sample_rate=22050, max_length=1000
    )

    # Sobrescrever os textos padrão
    audio_sampler.test_texts = test_texts

    print(f"🎯 Gerando amostras para {len(test_texts)} textos elaborados...")

    # Gerar amostras
    try:
        summary = audio_sampler.generate_samples(
            model=trainer.model,
            epoch=999,  # Época especial para samples finais
            text_preprocessor=text_preprocessor,
            device=device,
            num_samples_per_speaker=3,  # 3 textos por speaker
        )

        print(f"✅ Geração concluída!")
        print(f"📊 Total de amostras: {summary.get('total_samples', 0)}")
        print(f"📁 Diretório: {summary.get('output_directory', 'N/A')}")

        if summary.get("total_samples", 0) > 0:
            print(f"\n🎉 Sucesso! Amostras finais geradas.")
            print(f"🔊 Localização: {output_dir}/epoch_999/")
            print(f"\n📋 Resumo por speaker:")

            # Agrupar por speaker
            speakers = {}
            for sample in summary.get("samples", []):
                speaker = sample["speaker_name"]
                if speaker not in speakers:
                    speakers[speaker] = []
                speakers[speaker].append(sample)

            for speaker, samples in speakers.items():
                print(f"   🎤 {speaker}: {len(samples)} amostras")
                for i, sample in enumerate(samples):
                    duration = sample["metadata"]["audio_length_seconds"]
                    print(f"      {i+1}. {sample['text'][:50]}... ({duration:.1f}s)")

        else:
            print(f"\n❌ Erro: Nenhuma amostra foi gerada.")

    except Exception as e:
        print(f"❌ Erro na geração: {e}")
        import traceback

        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Geração de amostras finais")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/vits2_test/last-v4.ckpt",
        help="Caminho para o checkpoint mais recente",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/vits2_test_config.yaml",
        help="Caminho para a configuração",
    )
    parser.add_argument(
        "--output", type=str, default="final_samples", help="Diretório de saída"
    )

    args = parser.parse_args()

    # Verificar se arquivos existem
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint não encontrado: {args.checkpoint}")
        return

    if not Path(args.config).exists():
        print(f"❌ Config não encontrada: {args.config}")
        return

    generate_final_samples(args.checkpoint, args.config, args.output)


if __name__ == "__main__":
    main()
