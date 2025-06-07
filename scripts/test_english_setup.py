#!/usr/bin/env python3
"""
Script de Teste - Configuração para Inglês
==========================================

Verifica se o sistema ValeTTS está adequadamente configurado
para processar datasets em inglês.

Uso:
    python scripts/test_english_setup.py
"""

import os
import sys
from pathlib import Path

# Adicionar root ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))


def test_text_preprocessing():
    """Testa o preprocessamento de texto para inglês."""
    print("🧪 Testando preprocessamento de texto...")

    try:
        from valetts.data.preprocessing.text import create_text_preprocessor

        # Teste 1: Processador para inglês
        print("\n1️⃣ Testando processador para inglês:")
        en_processor = create_text_preprocessor("en-us", use_phonemes=True)

        test_texts = [
            "Hello world! This is a test.",
            "Dr. Smith has 123 books on 1st street.",
            "I can't believe it's working!",
            "The quick brown fox jumps.",
        ]

        for text in test_texts:
            normalized = en_processor.normalize_text(text)
            encoded = en_processor.encode(text)
            print(f"   Original: {text}")
            print(f"   Normalizado: {normalized}")
            print(f"   Encoded length: {len(encoded)}")
            print()

        print(f"✅ Vocabulário inglês: {en_processor.get_vocab_size()} tokens")

        # Teste 2: Processador para português (comparação)
        print("\n2️⃣ Testando processador para português:")
        pt_processor = create_text_preprocessor("pt-br", use_phonemes=False)

        text_pt = "Olá mundo! Este é um teste com Dr. Silva."
        normalized_pt = pt_processor.normalize_text(text_pt)
        encoded_pt = pt_processor.encode(text_pt)

        print(f"   Original: {text_pt}")
        print(f"   Normalizado: {normalized_pt}")
        print(f"   Encoded length: {len(encoded_pt)}")
        print(
            f"✅ Vocabulário português: {pt_processor.get_vocab_size()} tokens"
        )

        return True

    except Exception as e:
        print(f"❌ Erro no teste de preprocessamento: {e}")
        return False


def test_phonemizer():
    """Testa o phonemizer para inglês."""
    print("\n🔊 Testando phonemizer...")

    try:
        from phonemizer import phonemize

        test_text = "hello world this is a test"
        phonemes = phonemize(
            test_text,
            language="en-us",
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            with_stress=False,
        )

        print(f"   Texto: {test_text}")
        print(f"   Phonemes: {phonemes}")
        print("✅ Phonemizer funcionando!")
        return True

    except Exception as e:
        print(f"❌ Erro no phonemizer: {e}")
        print("   Verifique se espeak está instalado:")
        print("   sudo apt-get install espeak espeak-data")
        return False


def test_config_loading():
    """Testa carregamento da configuração para inglês."""
    print("\n⚙️ Testando configuração para inglês...")

    try:
        import yaml

        config_path = "configs/training/vits2_english_dataset.yaml"
        if not os.path.exists(config_path):
            print(f"❌ Configuração não encontrada: {config_path}")
            return False

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Verificar parâmetros específicos para inglês
        data_config = config.get("data", {})
        locale_column = data_config.get("locale_column", "unknown")
        text_processor = data_config.get("text_processor", {})
        dataset_config = config.get("dataset_config", {})
        expected_locale = dataset_config.get("expected_locale", "unknown")

        print(f"   Coluna de locale: {locale_column}")
        print(f"   Locale esperado: {expected_locale}")
        print(f"   Usar phonemes: {text_processor.get('use_phonemes', False)}")
        print(
            f"   Classe do processador: {text_processor.get('class', 'N/A')}"
        )

        if expected_locale.lower() in ["en", "en-us", "en-gb"]:
            print("✅ Configuração para inglês detectada!")
            return True
        else:
            print(f"⚠️ Configuração não está para inglês: {expected_locale}")
            return False

    except Exception as e:
        print(f"❌ Erro ao carregar configuração: {e}")
        return False


def test_dependencies():
    """Testa dependências necessárias."""
    print("\n📦 Testando dependências...")

    dependencies = [
        ("phonemizer", "phonemizer"),
        ("nltk", "nltk"),
        ("pytorch_lightning", "lightning"),
        ("torch", "torch"),
        ("yaml", "yaml"),
    ]

    all_ok = True
    for name, module in dependencies:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - não instalado")
            all_ok = False

    return all_ok


def test_training_compatibility():
    """Testa compatibilidade com script de treinamento."""
    print("\n🚀 Testando compatibilidade de treinamento...")

    try:
        # Verificar se o script de treinamento aceita configuração inglesa
        config_path = "configs/training/vits2_english_dataset.yaml"

        if not os.path.exists(config_path):
            print(f"❌ Configuração não encontrada: {config_path}")
            return False

        # Simular carregamento como no script de treinamento
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Verificar estrutura necessária
        required_sections = ["model", "training", "data", "hardware"]
        for section in required_sections:
            if section not in config:
                print(f"❌ Seção '{section}' faltando na configuração")
                return False

        # Verificar configurações específicas para inglês
        dataset_config = config.get("dataset_config", {})
        expected_locale = dataset_config.get("expected_locale", "")
        if expected_locale.lower() not in [
            "en",
            "en-us",
            "en-gb",
        ]:
            print("❌ Configuração não está para inglês")
            return False

        print("✅ Script de treinamento compatível!")
        return True

    except Exception as e:
        print(f"❌ Erro no teste de treinamento: {e}")
        return False


def main():
    """Função principal."""
    print("🔧 TESTE DE CONFIGURAÇÃO - VALETTS PARA INGLÊS")
    print("=" * 50)

    tests = [
        ("Preprocessamento de Texto", test_text_preprocessing),
        ("Phonemizer", test_phonemizer),
        ("Configuração", test_config_loading),
        ("Dependências", test_dependencies),
        ("Compatibilidade de Treinamento", test_training_compatibility),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n📋 Executando: {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))

    # Resumo final
    print("\n" + "=" * 50)
    print("📊 RESUMO DOS TESTES")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1

    print(f"\nResultado: {passed}/{total} testes passaram")

    if passed == total:
        print("\n🎉 SISTEMA PRONTO PARA DATASETS EM INGLÊS!")
        print("\nPara treinar com dataset em inglês:")
        print(
            "python scripts/train_vits2.py --config configs/training/vits2_english_dataset.yaml"
        )
    else:
        print(
            "\n⚠️ Alguns testes falharam. Corrija os problemas antes de continuar."
        )

    print("\n📝 Dicas:")
    print(
        "• Certifique-se que seu dataset CSV tem colunas: audio_path, text_normalized, speaker_id"
    )
    print(
        "• Ajuste o caminho do dataset em configs/training/vits2_english_dataset.yaml"
    )
    print(
        "• Para phonemes de qualidade, instale: sudo apt-get install espeak espeak-data"
    )

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
