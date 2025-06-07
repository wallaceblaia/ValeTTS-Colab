# Makefile para ValeTTS
# Comandos comuns de desenvolvimento

.PHONY: help install install-dev clean test lint format docs build deploy

# Configurações
PYTHON := python3
UV := uv

help: ## Mostra esta mensagem de ajuda
	@echo "ValeTTS - Comandos de Desenvolvimento"
	@echo ""
	@echo "Comandos disponíveis:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Instala o projeto
	$(UV) sync

install-dev: ## Instala o projeto com dependências de desenvolvimento
	$(UV) sync --extra dev
	$(UV) run pre-commit install

clean: ## Limpa arquivos temporários e cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .venv/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf bandit-report.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test: ## Executa todos os testes
	$(UV) run pytest tests/ -v

test-unit: ## Executa testes unitários
	$(UV) run pytest tests/unit/ -v

test-integration: ## Executa testes de integração
	$(UV) run pytest tests/integration/ -v

test-performance: ## Executa testes de performance
	$(UV) run pytest tests/performance/ -v --benchmark-only

test-cov: ## Executa testes com coverage
	$(UV) run pytest tests/ -v --cov=valetts --cov-report=html --cov-report=term

lint: ## Executa linting completo
	$(UV) run black --check .
	$(UV) run isort --check-only .
	$(UV) run flake8 .
	$(UV) run mypy valetts/
	$(UV) run bandit -r valetts/

format: ## Formata o código
	$(UV) run black .
	$(UV) run isort .

pre-commit: ## Executa pre-commit hooks
	$(UV) run pre-commit run --all-files

docs: ## Gera documentação
	$(UV) run sphinx-build -b html docs/ docs/_build/html

docs-serve: ## Serve documentação localmente
	$(UV) run python -m http.server 8000 --directory docs/_build/html

build: ## Faz build do pacote
	$(UV) build

build-wheel: ## Faz build apenas do wheel
	$(UV) build --wheel

install-local: build ## Instala versão local
	pip install dist/*.whl --force-reinstall

# Comandos de treinamento
train: ## Treina modelo base
	$(UV) run python scripts/train.py --config configs/training/vits2_training.yaml

train-dev: ## Treina modelo em modo desenvolvimento (menos épocas)
	$(UV) run python scripts/train.py --config configs/training/vits2_training.yaml \
		trainer.max_epochs=10 data.batch_size=4

# Comandos de inferência
infer: ## Executa inferência de exemplo
	$(UV) run python scripts/inference.py --text "Hello world" --output example.wav

infer-pt: ## Executa inferência em português
	$(UV) run python scripts/inference.py --text "Olá mundo" --language pt --output exemplo.wav

# Comandos de desenvolvimento
dev-setup: install-dev ## Setup completo de desenvolvimento
	@echo "✅ Ambiente de desenvolvimento configurado!"
	@echo "Execute 'make test' para verificar se tudo está funcionando"

dev-check: lint test ## Verifica se código está pronto para commit
	@echo "✅ Código pronto para commit!"

dev-reset: clean install-dev ## Reset completo do ambiente
	@echo "✅ Ambiente resetado!"

# Comandos de CI/CD
ci-local: lint test build ## Simula CI localmente
	@echo "✅ CI local executado com sucesso!"

release-check: ## Verifica se está pronto para release
	$(UV) run python -c "import valetts; print(f'Version: {valetts.__version__}')"
	make test
	make lint
	make build
	@echo "✅ Pronto para release!"

# Comandos de deployment
deploy-test: build ## Deploy para PyPI test
	$(UV) run twine upload --repository testpypi dist/*

deploy: build ## Deploy para PyPI
	$(UV) run twine upload dist/*

# Comandos de manutenção
update: ## Atualiza dependências
	$(UV) sync --upgrade

update-pre-commit: ## Atualiza pre-commit hooks
	$(UV) run pre-commit autoupdate

security-check: ## Verifica vulnerabilidades de segurança
	$(UV) run bandit -r valetts/ -f json -o bandit-report.json
	$(UV) run safety check

# Comandos de benchmark
benchmark: ## Executa benchmarks de performance
	$(UV) run python scripts/benchmark.py

profile: ## Faz profiling de performance
	$(UV) run python -m cProfile -o profile.stats scripts/inference.py --text "Profiling test"

# Comandos de dados
download-data: ## Download de datasets exemplo
	mkdir -p data/raw
	@echo "TODO: Implementar download de datasets"

preprocess-data: ## Preprocessa dados
	$(UV) run python scripts/preprocess_data.py

# Informações do sistema
info: ## Mostra informações do sistema
	@echo "Sistema: $$(uname -s)"
	@echo "Python: $$(python --version)"
	@echo "UV: $$(uv --version)"
	@echo "GPU: $$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'Não disponível')" 