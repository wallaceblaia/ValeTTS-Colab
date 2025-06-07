#!/bin/bash
# Script de setup de desenvolvimento para ValeTTS

set -e  # Exit on any error

echo "🚀 Configurando ambiente de desenvolvimento ValeTTS..."

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para prints coloridos
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar sistema operacional
print_status "Verificando sistema operacional..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    OS="windows"
else
    print_error "Sistema operacional não suportado: $OSTYPE"
    exit 1
fi
print_success "Sistema detectado: $OS"

# Verificar Python
print_status "Verificando Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
    print_success "Python encontrado: $PYTHON_VERSION"

    # Verificar versão mínima (3.10)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d '.' -f 1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d '.' -f 2)

    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        print_success "Versão do Python compatível (>=3.10)"
    else
        print_error "Python 3.10+ é necessário. Versão atual: $PYTHON_VERSION"
        print_status "Por favor, instale Python 3.10 ou superior"
        exit 1
    fi
else
    print_error "Python3 não encontrado"
    exit 1
fi

# Verificar/Instalar UV
print_status "Verificando UV..."
if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version)
    print_success "UV encontrado: $UV_VERSION"
else
    print_status "Instalando UV..."
    if [[ "$OS" == "windows" ]]; then
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi

    # Adicionar UV ao PATH para esta sessão
    export PATH="$HOME/.cargo/bin:$PATH"

    if command -v uv &> /dev/null; then
        print_success "UV instalado com sucesso"
    else
        print_error "Falha na instalação do UV"
        exit 1
    fi
fi

# Verificar Git
print_status "Verificando Git..."
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version)
    print_success "Git encontrado: $GIT_VERSION"
else
    print_error "Git não encontrado. Por favor, instale Git primeiro."
    exit 1
fi

# Verificar CUDA (opcional)
print_status "Verificando CUDA..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    print_success "NVIDIA GPU detectada, driver version: $CUDA_VERSION"
else
    print_warning "CUDA não detectado. O projeto funcionará apenas em CPU."
fi

# Instalar dependências
print_status "Instalando dependências do projeto..."
print_status "Instalando apenas dependências essenciais (sem TensorRT pesado)..."
uv sync --extra dev

if [ $? -eq 0 ]; then
    print_success "Dependências instaladas com sucesso"
else
    print_error "Falha na instalação das dependências"
    exit 1
fi

# Instalar pre-commit hooks
print_status "Configurando pre-commit hooks..."
uv run pre-commit install

if [ $? -eq 0 ]; then
    print_success "Pre-commit hooks configurados"
else
    print_warning "Falha na configuração dos pre-commit hooks"
fi

# Criar diretórios necessários
print_status "Criando diretórios necessários..."
mkdir -p data/{raw,processed,models}
mkdir -p logs
mkdir -p outputs
mkdir -p models
print_success "Diretórios criados"

# Verificar instalação
print_status "Verificando instalação..."

# Testar import do projeto
if uv run python -c "import valetts; print('ValeTTS version:', valetts.__version__)" &> /dev/null; then
    print_success "Projeto pode ser importado corretamente"
else
    print_warning "Problema ao importar o projeto (normal se ainda não implementado)"
fi

# Executar testes
print_status "Executando testes de exemplo..."
if uv run pytest tests/unit/test_example.py -v; then
    print_success "Testes executaram com sucesso"
else
    print_warning "Alguns testes falharam"
fi

# Executar linting
print_status "Verificando qualidade do código..."
if uv run pre-commit run --all-files; then
    print_success "Código está em conformidade com os padrões"
else
    print_warning "Alguns problemas de qualidade de código detectados"
fi

# Criar arquivo .env de exemplo
print_status "Criando arquivo .env de exemplo..."
cat > .env.example << EOF
# Configurações de desenvolvimento
WANDB_API_KEY=your_wandb_api_key_here
CUDA_VISIBLE_DEVICES=0
PYTHONPATH=.

# Configurações de database (opcional)
DATABASE_URL=postgresql://valetts:valetts@localhost:5432/valetts
REDIS_URL=redis://localhost:6379

# Configurações de API
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=true
EOF
print_success "Arquivo .env.example criado"

# Informações finais
echo ""
echo "🎉 Setup de desenvolvimento concluído!"
echo ""
echo "📋 Próximos passos:"
echo "   1. Copie .env.example para .env e configure as variáveis"
echo "   2. Execute 'make help' para ver comandos disponíveis"
echo "   3. Execute 'make test' para verificar se tudo está funcionando"
echo "   4. Execute 'make train-dev' para treinar um modelo de exemplo"
echo ""
echo "🛠️  Comandos úteis:"
echo "   • make dev-setup     - Reconfigure o ambiente"
echo "   • make test          - Execute todos os testes"
echo "   • make lint          - Verifique qualidade do código"
echo "   • make format        - Formate o código"
echo "   • make dev-check     - Verificação completa antes de commit"
echo ""
echo "🐳 Docker (opcional):"
echo "   • docker-compose up valetts-dev  - Ambiente Docker"
echo "   • docker-compose up jupyter      - Jupyter Lab"
echo "   • docker-compose up tensorboard  - TensorBoard"
echo ""
echo "📚 Documentação:"
echo "   • README.md - Documentação principal"
echo "   • docs/ - Documentação detalhada"
echo "   • PLANO_DE_TRABALHO.md - Cronograma do projeto"
echo ""

# Verificar se o ambiente virtual está ativado
if [[ "$VIRTUAL_ENV" != *"valetts"* ]]; then
    print_warning "Para ativar o ambiente virtual:"
    echo "   source .venv/bin/activate"
fi

print_success "Ambiente de desenvolvimento configurado com sucesso! 🚀"
