#!/usr/bin/env python3
"""
Teste Simples de Treinamento ValeTTS
===================================

Script básico para testar se o sistema de treinamento está funcionando.
"""

import os
import sys
from pathlib import Path

import yaml

# Adicionar o diretório raiz ao path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))


def test_config_loading():
    """Testa carregamento da configuração."""
    print("🔧 Testando carregamento de configuração...")

    config_path = "configs/training/vits2_dataset_unificado.yaml"

    if not os.path.exists(config_path):
        print(f"❌ Arquivo de configuração não encontrado: {config_path}")
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        print(f"✅ Configuração carregada: {config_path}")
        print(f"   • Modelo: {config['model']['name']}")
        print(f"   • Dataset: {config['data']['data_dir']}")
        print(f"   • Épocas: {config['training']['max_epochs']}")
        print(f"   • Batch size: {config['training']['batch_size']}")
        return True

    except Exception as e:
        print(f"❌ Erro ao carregar configuração: {e}")
        return False


def test_dataset_path():
    """Testa se o dataset existe."""
    print("\n📊 Testando dataset...")

    dataset_path = "data/generated/Dataset-Unificado"
    metadata_path = os.path.join(dataset_path, "metadata.csv")

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset não encontrado: {dataset_path}")
        return False

    if not os.path.exists(metadata_path):
        print(f"❌ Metadata não encontrada: {metadata_path}")
        return False

    print(f"✅ Dataset encontrado: {dataset_path}")
    print(f"✅ Metadata encontrada: {metadata_path}")

    # Contar linhas no metadata
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            lines = sum(1 for line in f) - 1  # -1 para header
        print(f"   • Amostras no dataset: {lines:,}")
        return True
    except Exception as e:
        print(f"❌ Erro ao ler metadata: {e}")
        return False


def test_pytorch():
    """Testa PyTorch e GPU."""
    print("\n🔥 Testando PyTorch...")

    try:
        import torch

        print(f"✅ PyTorch: {torch.__version__}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU: {gpu_name}")
            print(f"✅ VRAM: {vram_gb:.1f} GB")
        else:
            print("⚠️ CUDA não disponível - usando CPU")

        return True

    except ImportError:
        print("❌ PyTorch não instalado")
        return False


def test_dependencies():
    """Testa dependências críticas."""
    print("\n📦 Testando dependências...")

    deps = [
        "pytorch_lightning",
        "librosa",
        "soundfile",
        "numpy",
        "scipy",
        "tensorboard",
    ]

    all_ok = True
    for dep in deps:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - NÃO ENCONTRADO")
            all_ok = False

    return all_ok


def test_imports():
    """Testa imports do ValeTTS (básicos)."""
    print("\n🧪 Testando imports básicos...")

    try:
        # Teste muito básico
        import valetts

        print(f"✅ valetts v{valetts.__version__}")
        return True

    except Exception as e:
        print(f"❌ Erro nos imports do ValeTTS: {e}")
        return False


def main():
    """Executa todos os testes."""
    print("🚀 TESTE SIMPLES DE TREINAMENTO ValeTTS")
    print("=" * 50)

    tests = [
        ("Configuração", test_config_loading),
        ("Dataset", test_dataset_path),
        ("PyTorch", test_pytorch),
        ("Dependências", test_dependencies),
        ("Imports ValeTTS", test_imports),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Erro no teste {name}: {e}")
            results.append((name, False))

    # Resumo
    print("\n" + "=" * 50)
    print("📋 RESUMO DOS TESTES:")

    passed = 0
    for name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"   {name}: {status}")
        if result:
            passed += 1

    print(f"\n🎯 Total: {passed}/{len(results)} testes passaram")

    if passed == len(results):
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("💡 Sistema pronto para treinamento!")
    else:
        print("\n⚠️ ALGUNS TESTES FALHARAM")
        print("💡 Corrija os problemas antes de prosseguir")

    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
