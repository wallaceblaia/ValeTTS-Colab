# 🛠️ Guia de Solução de Problemas - Google Colab

## ❌ Problemas Comuns e Soluções

### 1. Erro: "No such file or directory: 'requirements.txt'"

**Problema**: O notebook não consegue encontrar os arquivos do projeto.

**Solução**:
```python
# Execute esta célula para verificar:
import os
print(f"📁 Diretório atual: {os.getcwd()}")
!ls -la
```

**Se o diretório estiver incorreto**:
```python
# Navegue para o diretório correto:
%cd /content/ValeTTS
!ls -la requirements.txt setup.py
```

### 2. Erro: "git clone failed" ou repositório não encontrado

**Problema**: URL do repositório incorreta ou repositório privado.

**Solução**: Use o link correto:
```bash
!git clone https://github.com/wallaceblaia/ValeTTS-Colab.git ValeTTS
```

### 3. Erro: "ModuleNotFoundError: No module named 'valetts'"

**Problema**: Módulo não instalado ou não no PYTHONPATH.

**Solução**:
```python
# Adicionar ao PYTHONPATH:
import sys
import os
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# Verificar importação:
import valetts
print("✅ Módulo valetts importado com sucesso")
```

### 4. Dataset não encontrado

**Problema**: `Dataset-Unificado.tar.gz` não está no local correto.

**Verificação**:
```python
import os
dataset_path = "/content/drive/MyDrive/ValeTTS-Colab/Dataset-Unificado.tar.gz"
print(f"Dataset existe: {os.path.exists(dataset_path)}")

# Se não existir, listar conteúdo do diretório:
!ls -la "/content/drive/MyDrive/ValeTTS-Colab/"
```

**Soluções**:
1. **Upload manual**: Faça upload do dataset para `MeuDrive/ValeTTS-Colab/`
2. **Caminho diferente**: Atualize o caminho na célula:
   ```python
   dataset_path = "/content/drive/MyDrive/SEU_CAMINHO_AQUI/Dataset-Unificado.tar.gz"
   ```

### 5. Erro de permissão ou autenticação

**Problema**: Erro ao acessar Google Drive.

**Solução**:
```python
# Reconectar ao Drive:
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### 6. GPU não disponível

**Verificação**:
```python
import torch
print(f"CUDA disponível: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nenhuma'}")
```

**Solução**:
- Vá em `Runtime` → `Change runtime type` → `Hardware accelerator` → `GPU`

### 7. Erro de import lightning

**Problema**: Versão incorreta do PyTorch Lightning.

**Solução**:
```python
# Instalar versão correta:
!pip install pytorch-lightning==2.1.0 --quiet
```

## 🆘 Comandos de Emergência

### Reset Completo
```python
# Limpar tudo e começar novamente:
%cd /content
!rm -rf ValeTTS
!git clone https://github.com/wallaceblaia/ValeTTS-Colab.git ValeTTS
%cd ValeTTS
```

### Verificação Completa do Sistema
```python
import os
import sys

print("🔍 DIAGNÓSTICO COMPLETO")
print("=" * 50)
print(f"📁 Diretório atual: {os.getcwd()}")
print(f"🐍 Python path: {sys.path[:3]}...")

# Verificar arquivos essenciais
files = ['requirements.txt', 'setup.py', 'scripts/train_vits2.py']
for file in files:
    status = "✅" if os.path.exists(file) else "❌"
    print(f"{status} {file}")

# Verificar GPU
import torch
print(f"🎮 CUDA: {torch.cuda.is_available()}")

# Verificar Drive
drive_connected = os.path.exists('/content/drive/MyDrive')
print(f"💾 Drive: {'✅' if drive_connected else '❌'}")
```

### Instalação Manual de Dependências
```python
# Se requirements.txt falhar:
!pip install torch torchaudio pytorch-lightning tensorboard pyyaml matplotlib scipy librosa numpy pandas jupyter

# Verificar instalação:
import torch
import pytorch_lightning
import yaml
print("✅ Dependências básicas instaladas")
```

## 📱 Links Úteis

### Acesso Direto ao Notebook
```
https://colab.research.google.com/github/wallaceblaia/ValeTTS-Colab/blob/main/colab_training_vits2.ipynb
```

### Repositório GitHub
```
https://github.com/wallaceblaia/ValeTTS-Colab
```

### Verificar Status do Repositório
```python
!git remote -v
!git status
!git log --oneline -5
```

## 💡 Dicas de Performance

### Otimizar para GPU
```python
# Verificar uso de GPU durante treinamento:
!nvidia-smi -l 1  # Atualiza a cada segundo
```

### Monitorar Recursos
```python
# Verificar RAM:
!free -h

# Verificar disco:
!df -h
```

### Backup Automático
```python
# Configurar backup automático a cada época:
backup_dir = "/content/drive/MyDrive/ValeTTS-Colab/training_backup"
!mkdir -p {backup_dir}
print(f"📁 Backup configurado em: {backup_dir}")
```

---

**🆘 Se nada funcionar**: Reinicie o runtime (`Runtime` → `Restart runtime`) e execute o notebook desde o início.
