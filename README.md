# 🎙️ ValeTTS - Sistema TTS Híbrido Avançado

**Sistema de Text-to-Speech com arquitetura híbrida: VITS2 + Meta-Learning + BigVGAN-v2 + Controle Prosódico**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wallacemt/ValeTTS/blob/main/colab_training_vits2.ipynb)

## 🚀 Início Rápido no Google Colab

**Treinamento com performance 25-200x mais rápida que CPU local!**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wallacemt/ValeTTS/blob/main/colab_training_vits2.ipynb)

1. **Clique no botão acima** para abrir no Google Colab
2. **Configure sua API OpenRouter** (opcional, para monitor LLM)
3. **Execute todas as células** - o sistema fará tudo automaticamente
4. **Aguarde o treinamento** - monitoramento via TensorBoard integrado

## 🏗️ Arquitetura

### 🎯 Componentes Principais

- **🎵 VITS2**: Modelo base de síntese neural end-to-end
- **🧠 Meta-Learning (MAML)**: Adaptação rápida para novos speakers
- **🔊 BigVGAN-v2**: Vocoder neural de alta qualidade
- **🎭 Controle Prosódico**: Manipulação de entonação, ritmo e emoção
- **🤖 Monitor LLM**: Otimização inteligente durante treinamento

### 📊 Performance

| Ambiente | GPU | Performance | Speedup vs Local |
|----------|-----|-------------|------------------|
| **Local** | CPU | 0.08 it/s | 1x (baseline) |
| **Colab** | T4 | 1-2 it/s | 25-50x |
| **Colab Pro** | P100 | 2-3 it/s | 50-75x |
| **Colab Pro+** | V100 | 3-5 it/s | 75-125x |
| **Colab Pro+** | A100 | 5-8 it/s | 125-200x |

## 🛠️ Instalação Local

### Pré-requisitos

```bash
# Python 3.8+
python --version

# Git
git --version
```

### Setup

```bash
# Clone do repositório
git clone https://github.com/wallacemt/ValeTTS.git
cd ValeTTS

# Instalar dependências
pip install -r requirements.txt

# Instalar projeto
pip install -e .
```

## 🎯 Uso

### 🚀 Treinamento Local

```bash
# Comando básico
python scripts/train_vits2.py --config configs/training/vits2_dataset_unificado.yaml

# Com monitor LLM (configure OPENROUTER_API_KEY)
export OPENROUTER_API_KEY="sua_api_key"
python scripts/train_vits2.py --config configs/training/vits2_dataset_unificado.yaml

# Resumir treinamento
python scripts/train_vits2.py --config configs/training/vits2_dataset_unificado.yaml --resume logs/checkpoints/last.ckpt
```

### 🎙️ Inferência

```bash
# Síntese básica
python scripts/inference.py --text "Olá, mundo!" --speaker_id 0 --output audio_output.wav

# Com controle prosódico
python scripts/inference.py --text "Olá, mundo!" --speaker_id 0 --emotion happy --speed 1.2 --pitch 1.1
```

## 📁 Estrutura do Projeto

```
ValeTTS/
├── 📓 colab_training_vits2.ipynb    # Notebook para Google Colab
├── 🐍 valetts/                      # Módulo principal
│   ├── 🎵 models/                   # Modelos (VITS2, BigVGAN, etc.)
│   ├── 📊 data/                     # Datasets e loaders
│   ├── 🏋️ training/                 # Sistema de treinamento
│   ├── 🎙️ inference/               # Sistema de inferência
│   ├── 📈 evaluation/               # Métricas e avaliação
│   └── 🛠️ utils/                    # Utilitários
├── 🔧 scripts/                      # Scripts de treinamento/análise
├── ⚙️ configs/                      # Configurações YAML
└── 📊 data/                         # Datasets
```

## 📋 Configuração

### 🎯 Dataset

O sistema usa o **Dataset-Unificado** com:
- **📊 Tamanho**: ~2.5GB
- **🎤 Samples**: ~23.000 amostras
- **👥 Speakers**: 52 speakers únicos
- **🌍 Idioma**: Português brasileiro

### 🔧 Treinamento

```yaml
# configs/training/vits2_dataset_unificado.yaml
model:
  num_speakers: 52

training:
  epochs: 200
  batch_size: 12
  learning_rate: 2e-4
  mixed_precision: true

llm_monitor:
  enabled: true
  provider: "openrouter"
  model: "anthropic/claude-3-5-sonnet-20241022"
  monitor_every_epochs: 5
```

## 🤖 Monitor LLM

### 🧠 Funcionalidades

- **📊 Análise inteligente**: Avalia métricas a cada 5 épocas
- **⚙️ Ajustes automáticos**: Learning rate, batch size, scheduler
- **🛡️ Aprovação humana**: Para mudanças críticas (>30%)
- **📈 Histórico**: Salva análises e decisões
- **🔄 Configuração dinâmica**: Atualiza parâmetros em tempo real

### 🔑 Configuração API

```bash
# OpenRouter (recomendado)
export OPENROUTER_API_KEY="sua_api_key"

# Ou configure no Colab
OPENROUTER_API_KEY = "sua_api_key"  # @param {type:"string"}
```

## 📊 Monitoramento

### 📈 TensorBoard

```bash
# Local
tensorboard --logdir logs/tensorboard/

# Colab (automático)
%tensorboard --logdir logs/tensorboard/
```

### 📋 Métricas

- **🎵 Loss de Reconstrução**: Qualidade da síntese
- **🎭 Loss de Adversarial**: Realismo do áudio
- **🎯 Loss de Duração**: Precisão temporal
- **📊 MOS Score**: Qualidade perceptual
- **⚡ Velocidade**: Iterações por segundo

## 🎯 Roadmap

### ✅ Concluído

- [x] Sistema VITS2 multi-speaker
- [x] Monitor LLM inteligente
- [x] Notebook Google Colab
- [x] Dataset unificado
- [x] Sistema de treinamento robusto

### 🚧 Em Desenvolvimento

- [ ] Interface web para inferência
- [ ] API REST para produção
- [ ] Suporte a múltiplos idiomas
- [ ] Fine-tuning com poucos dados
- [ ] Controle prosódico avançado

### 🔮 Futuro

- [ ] Síntese em tempo real
- [ ] Clonagem de voz zero-shot
- [ ] Integração com streaming
- [ ] Mobile deployment
- [ ] Plugin para DAWs

## 🤝 Contribuição

1. **Fork** o repositório
2. **Crie** uma branch: `git checkout -b feature/nova-funcionalidade`
3. **Commit** suas mudanças: `git commit -m 'Adiciona nova funcionalidade'`
4. **Push** para a branch: `git push origin feature/nova-funcionalidade`
5. **Abra** um Pull Request

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- **VITS2**: [Original Paper](https://arxiv.org/abs/2307.16430)
- **BigVGAN**: [Original Repository](https://github.com/NVIDIA/BigVGAN)
- **PyTorch Lightning**: Framework de treinamento
- **Google Colab**: Plataforma de desenvolvimento

---

**🎉 ValeTTS - Transformando texto em fala com IA de última geração!**

[![GitHub stars](https://img.shields.io/github/stars/wallacemt/ValeTTS?style=social)](https://github.com/wallacemt/ValeTTS/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/wallacemt/ValeTTS?style=social)](https://github.com/wallacemt/ValeTTS/network/members)
