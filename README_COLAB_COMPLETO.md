# 🚀 ValeTTS - Guia Completo Google Colab Pro

## 📋 **MIGRAÇÃO CONCLUÍDA!**

✅ **Setup completo preparado para Google Colab Pro**
✅ **Código no GitHub**: https://github.com/wallaceblaia/ValeTTS-Colab
✅ **Dataset compactado**: `Dataset-Unificado.tar.gz` (2.5GB)
✅ **Notebooks otimizados** com OpenRouter + TensorBoard

---

## 🎯 **NOTEBOOKS DISPONÍVEIS**

### 📄 **1. Setup Inicial**
- **Arquivo**: `01_setup_colab.ipynb`
- **Função**: Verificação de GPU, instalação de dependências

### 📄 **2. Treinamento Principal** ⭐
- **Arquivo**: `02_train_vits2_openrouter.ipynb`
- **Função**: Treinamento completo com monitoramento LLM

### 📄 **3. Monitoramento Avançado**
- **Arquivo**: `03_monitoring_tensorboard.ipynb`
- **Função**: TensorBoard, relatórios LLM, visualização

---

## 🔧 **CONFIGURAÇÃO AUTOMÁTICA POR GPU**

| GPU | VRAM | Batch Size | Performance Esperada |
|-----|------|------------|---------------------|
| **Tesla T4** | 15GB | 6 | 1-2 it/s |
| **Tesla P100** | 16GB | 8 | 2-3 it/s |
| **Tesla V100** | 16GB | 10 | 3-5 it/s |

**Otimizações aplicadas:**
- Mixed Precision (FP16)
- Gradient Accumulation 4x
- Tensor Cores ativados
- Modelo reduzido para cloud

---

## 🤖 **MONITORAMENTO LLM - OPENROUTER**

### 📋 **Recursos Disponíveis:**
- ✅ **Relatórios automáticos** a cada 5 épocas
- ✅ **Análise inteligente** de métricas de treino
- ✅ **Explicações educativas** sobre o progresso
- ✅ **Salvamento automático** no Google Drive

### 🔑 **Configuração da API Key:**
1. **Acesse**: https://openrouter.ai/keys
2. **Crie uma conta** (se não tiver)
3. **Gere uma API key**
4. **Configure no notebook** (célula 4):
   ```python
   OPENROUTER_API_KEY = "sk-or-v1-sua-key-aqui"
   ```

### 🧠 **Modelos Recomendados:**
- `anthropic/claude-3.5-sonnet` (melhor qualidade)
- `openai/gpt-4o` (alternativa rápida)
- `meta-llama/llama-3.1-8b-instruct` (economia)

### 💰 **Custos Estimados:**
- **Claude 3.5 Sonnet**: ~$0.20 por treino completo
- **GPT-4**: ~$0.15 por treino completo
- **Llama 3.1**: ~$0.05 por treino completo

---

## 📊 **TENSORBOARD - VISUALIZAÇÃO**

### 🎯 **Métricas Monitoradas:**
- **Loss do Gerador** (mel_loss, kl_loss, fm_loss)
- **Loss do Discriminador** (disc_loss)
- **Learning Rate** (otimização adaptiva)
- **Gradient Norm** (estabilidade do treino)
- **Validation Metrics** (overfitting)

### 🔍 **Acesso ao TensorBoard:**
- **URL**: http://localhost:6006
- **Atualização**: Tempo real
- **Persistência**: Salvo no Google Drive

---

## 📁 **ESTRUTURA NO GOOGLE DRIVE**

```
📁 /content/drive/MyDrive/ValeTTS/
├── 📄 Dataset-Unificado.tar.gz     # Dataset original (2.5GB)
├── 📁 checkpoints/                 # Modelos salvos (.ckpt)
├── 📁 logs/                        # TensorBoard logs
├── 📁 samples/                     # Amostras de áudio geradas
└── 📁 reports/                     # Relatórios LLM (.md)
```

---

## 🚀 **PASSO A PASSO - EXECUÇÃO**

### **ETAPA 1: Upload do Dataset**
1. **Abra**: https://drive.google.com
2. **Crie pasta**: `ValeTTS`
3. **Faça upload**: `Dataset-Unificado.tar.gz` ➔ `/ValeTTS/`

### **ETAPA 2: Abrir Google Colab**
1. **Acesse**: https://colab.research.google.com
2. **Escolha**: Google Colab Pro (recomendado)
3. **Upload notebooks**: Da pasta `colab_migration/notebooks/`

### **ETAPA 3: Configurar OpenRouter (Opcional)**
1. **Crie conta**: https://openrouter.ai
2. **Gere API key**: https://openrouter.ai/keys
3. **Configure no notebook**: Célula 4

### **ETAPA 4: Executar Treinamento**
1. **Execute**: `02_train_vits2_openrouter.ipynb`
2. **Monitore**: TensorBoard + Relatórios LLM
3. **Aguarde**: 8-15 horas (200 épocas)

### **ETAPA 5: Monitoramento (Paralelo)**
1. **Abra**: `03_monitoring_tensorboard.ipynb`
2. **Visualize**: Métricas em tempo real
3. **Escute**: Amostras geradas

---

## ⚡ **PERFORMANCE ESPERADA**

### 🎯 **Comparação com Local:**
- **Local (RTX 4090)**: 0.08 it/s (impraticável)
- **Colab T4**: 1-2 it/s (**25x mais rápido**)
- **Colab P100**: 2-3 it/s (**37x mais rápido**)
- **Colab V100**: 3-5 it/s (**62x mais rápido**)

### ⏱️ **Tempo Estimado:**
- **200 épocas**: 8-15 horas
- **Checkpoint**: A cada 10 épocas (~45-90 min)
- **Relatório LLM**: A cada 5 épocas (~22-45 min)
- **Amostras**: A cada 10 épocas

### 💾 **Uso de Recursos:**
- **VRAM**: 70-85% da GPU disponível
- **RAM**: ~8-12GB
- **Disk**: ~3-5GB (temporário)
- **Drive**: ~10-15GB (persistente)

---

## 🔧 **CONFIGURAÇÕES AVANÇADAS**

### 📊 **Ajustar Batch Size:**
```python
# No notebook, célula 5:
batch_size = 6  # T4: 6, P100: 8, V100: 10
```

### 🤖 **Desabilitar LLM Monitoring:**
```python
# No notebook, célula 4:
ENABLE_LLM_MONITORING = False
```

### 📈 **Modificar Frequência de Reports:**
```python
# No arquivo YAML gerado:
llm_monitoring:
  monitor_every_n_epochs: 3  # Default: 5
```

---

## 🆘 **RESOLUÇÃO DE PROBLEMAS**

### ❌ **"Dataset não encontrado"**
- Verificar path: `/content/drive/MyDrive/ValeTTS/Dataset-Unificado.tar.gz`
- Re-upload do arquivo
- Verificar conectividade do Drive

### ❌ **"Out of Memory (OOM)"**
- Reduzir batch_size na célula 5
- Reiniciar runtime: Runtime → Restart
- Trocar para GPU mais potente

### ❌ **"OpenRouter API Error"**
- Verificar API key válida
- Verificar saldo na conta OpenRouter
- Desabilitar LLM monitoring temporariamente

### ❌ **"TensorBoard não carrega"**
- Aguardar 30-60 segundos
- Refresh da página
- Verificar se processo está rodando

---

## 🎉 **RESULTADOS ESPERADOS**

### 🎵 **Qualidade do Áudio:**
- **Épocas 1-50**: Ruído, inteligibilidade baixa
- **Épocas 50-100**: Voz reconhecível, pronúncia melhorando
- **Épocas 100-150**: Boa qualidade, prosódia natural
- **Épocas 150-200**: Qualidade profissional

### 📊 **Métricas de Sucesso:**
- **Mel Loss**: < 2.0 (bom), < 1.5 (excelente)
- **KL Loss**: Estável após época 20
- **Discriminator Loss**: Balanceado com Generator

### 🎯 **Checkpoints Importantes:**
- **Época 50**: Primeiro teste de qualidade
- **Época 100**: Checkpoint para produção básica
- **Época 150**: Modelo robusto
- **Época 200**: Modelo final

---

## 🔗 **LINKS ÚTEIS**

- **🐙 GitHub**: https://github.com/wallaceblaia/ValeTTS-Colab
- **🤖 OpenRouter**: https://openrouter.ai
- **📊 Google Colab**: https://colab.research.google.com
- **💾 Google Drive**: https://drive.google.com
- **📈 TensorBoard**: Disponível no notebook

---

## 📞 **SUPORTE**

### 🤖 **Monitoramento Automático:**
Os relatórios LLM fornecerão análises detalhadas e sugestões de otimização durante o treinamento.

### 📊 **Métricas em Tempo Real:**
Use o TensorBoard para acompanhar o progresso e identificar problemas precocemente.

### 🎯 **Resultado Final:**
Após 200 épocas, você terá um modelo VITS2 treinado para síntese de voz em português brasileiro com qualidade profissional!

---

**🚀 BOA SORTE COM O TREINAMENTO! 🎙️**
