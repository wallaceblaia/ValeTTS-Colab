#!/usr/bin/env python3
"""
Script de Treinamento VITS2 com Monitoramento LLM
=================================================

Sistema revolucionário de treinamento TTS com análise assistida por IA.

Funcionalidades:
- Treinamento VITS2 completo
- Monitoramento LLM opcional (Claude 4 Sonnet)
- Otimização automática de hiperparâmetros
- Análise inteligente de progresso
- Suporte multi-GPU com Lightning

Uso:
    python scripts/train_vits2.py --config configs/training/vits2_config.yaml

Para usar monitoramento LLM:
    export OPENROUTER_API_KEY="sua_chave_openrouter"
    python scripts/train_vits2.py --config configs/training/vits2_config.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as L
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# Adicionar o diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from valetts.data.datasets.tts_dataset import TTSDataset
from valetts.data.loaders.tts import TTSDataLoader
from valetts.models.vits2 import VITS2
from valetts.training.monitoring import LLMMonitorConfig
from valetts.training.trainers.vits2 import VITS2Trainer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/training.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Carrega configuração do arquivo YAML."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        logger.info(f"✅ Configuração carregada: {config_path}")
        return config

    except Exception as e:
        logger.error(f"❌ Erro ao carregar configuração: {e}")
        raise


def create_llm_monitor_config(
    config: Dict[str, Any],
) -> Optional[LLMMonitorConfig]:
    """Cria configuração do monitoramento LLM."""
    llm_config = config.get("llm_monitor", {})

    if not llm_config.get("enabled", False):
        logger.info("ℹ️ Monitoramento LLM desabilitado")
        return None

    # Verificar API key
    api_key_env = llm_config.get("api_key_env", "OPENROUTER_API_KEY")
    if not os.getenv(api_key_env):
        logger.warning(f"⚠️ {api_key_env} não configurada - Desabilitando LLM")
        return None

    try:
        monitor_config = LLMMonitorConfig(
            enabled=True,
            provider=llm_config.get("provider", "openrouter"),
            model=llm_config.get(
                "model", "anthropic/claude-3-5-sonnet-20241022"
            ),
            api_key_env=api_key_env,
            base_url=llm_config.get(
                "base_url", "https://openrouter.ai/api/v1"
            ),
            # Frequência
            monitor_every_epochs=llm_config.get("monitor_every_epochs", 5),
            monitor_every_steps=llm_config.get("monitor_every_steps"),
            # Configuração dinâmica
            dynamic_config_path=llm_config.get(
                "dynamic_config_path",
                "configs/training/llm_dynamic_config.yaml",
            ),
            backup_original_config=llm_config.get(
                "backup_original_config", True
            ),
            # Ranges seguros
            lr_min=llm_config.get("lr_min", 1e-6),
            lr_max=llm_config.get("lr_max", 1e-2),
            lr_change_factor_max=llm_config.get("lr_change_factor_max", 2.0),
            batch_size_min=llm_config.get("batch_size_min", 4),
            batch_size_max=llm_config.get("batch_size_max", 128),
            batch_size_change_factor_max=llm_config.get(
                "batch_size_change_factor_max", 2.0
            ),
            loss_weight_min=llm_config.get("loss_weight_min", 0.1),
            loss_weight_max=llm_config.get("loss_weight_max", 50.0),
            loss_weight_change_factor_max=llm_config.get(
                "loss_weight_change_factor_max", 1.5
            ),
            # Segurança
            require_human_approval=llm_config.get(
                "require_human_approval", True
            ),
            critical_change_threshold=llm_config.get(
                "critical_change_threshold", 0.3
            ),
            max_consecutive_changes=llm_config.get(
                "max_consecutive_changes", 3
            ),
            # Histórico
            save_analysis_history=llm_config.get(
                "save_analysis_history", True
            ),
            max_history_entries=llm_config.get("max_history_entries", 50),
            include_context_epochs=llm_config.get("include_context_epochs", 3),
        )

        logger.info("🤖 Monitoramento LLM configurado com sucesso!")
        return monitor_config

    except Exception as e:
        logger.error(f"❌ Erro ao configurar monitoramento LLM: {e}")
        return None


def create_model(config: Dict[str, Any]) -> VITS2:
    """Cria o modelo VITS2."""
    from valetts.models.vits2.config import VITS2Config

    # Criar configuração do modelo
    model_config = config["model"]
    data_config = config["data"]

    vits2_config = VITS2Config(
        # Parâmetros do modelo
        text_encoder_hidden_dim=model_config.get(
            "text_encoder_hidden_dim", 192
        ),
        latent_dim=model_config.get("latent_dim", 192),
        mel_channels=model_config.get("mel_channels", 80),
        n_speakers=model_config.get("n_speakers", 1),
        speaker_embedding_dim=model_config.get("speaker_embedding_dim", 256),
        inference_only=model_config.get("inference_only", False),
        # Parâmetros do generator
        generator_initial_channels=model_config.get(
            "generator_initial_channels", 512
        ),
        decoder_hidden_dim=model_config.get("decoder_hidden_dim", 512),
        # Parâmetros de áudio
        sample_rate=data_config.get("sample_rate", 22050),
        n_mels=data_config.get("n_mels", 80),
        n_fft=data_config.get("n_fft", 1024),
        hop_length=data_config.get("hop_length", 256),
        win_length=data_config.get("win_length", 1024),
    )

    model = VITS2(vits2_config)

    logger.info(
        f"✅ Modelo VITS2 criado - Parâmetros: {sum(p.numel() for p in model.parameters()):,}"
    )
    return model


def create_data_loaders(config: Dict[str, Any]) -> tuple:
    """Cria data loaders de treino e validação."""
    data_config = config["data"]

    # Detectar idioma do dataset baseado na configuração ou coluna locale
    locale_column = data_config.get("locale_column")
    dataset_config = config.get("dataset_config", {})
    expected_locale = dataset_config.get("expected_locale")

    # Usar expected_locale se disponível, senão usar padrão
    if expected_locale:
        language = expected_locale
        if expected_locale == "en":
            language = "en-us"  # Converter para formato completo
    else:
        language = data_config.get("language", "pt-br")

    # Configurar processador de texto baseado no idioma
    text_processor_config = None
    if language.lower() in ["en", "en-us", "en-gb", "english"]:
        logger.info(f"🌍 Dataset em inglês detectado: {language}")
        text_processor_config = {
            "language": language,
            "use_phonemes": data_config.get("text_processor", {}).get(
                "use_phonemes", True
            ),
            "normalize_numbers": True,
            "normalize_whitespace": True,
            "lowercase": True,
        }
        logger.info(
            "📝 Usando processamento específico para inglês com phonemes"
        )
    else:
        logger.info(f"🌍 Dataset em português detectado: {language}")
        text_processor_config = {
            "language": language,
            "use_phonemes": False,  # Padrão para português
            "normalize_numbers": True,
            "normalize_whitespace": True,
            "lowercase": True,
        }

    # Configuração de áudio para os processadores
    audio_config = {
        "sample_rate": data_config.get("sample_rate", 22050),
        "n_mels": data_config.get("n_mels", 80),
        "n_fft": data_config.get("n_fft", 1024),
        "hop_length": data_config.get("hop_length", 256),
    }

    # Parâmetros para debug
    max_samples_debug = data_config.get("max_samples_debug")

    # Dataset de treino
    train_dataset = TTSDataset(
        data_dir=data_config["data_dir"],
        split="train",
        audio_config=audio_config,
        max_samples=max_samples_debug,
    )

    # Dataset de validação
    val_dataset = TTSDataset(
        data_dir=data_config["data_dir"],
        split="val",
        audio_config=audio_config,
        max_samples=max_samples_debug // 10 if max_samples_debug else None,
    )

    # Importar collate_fn do nosso dataset
    from torch.utils.data import DataLoader

    from valetts.data.datasets.tts_dataset import collate_fn

    # Data loaders usando DataLoader padrão com nosso collate_fn
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        collate_fn=collate_fn,
    )

    logger.info(
        f"✅ Data loaders criados - Train: {len(train_dataset)}, Val: {len(val_dataset)}"
    )
    return train_loader, val_loader


def create_callbacks(config: Dict[str, Any]) -> list:
    """Cria callbacks para o treinamento."""
    callbacks = []

    # Checkpoint callback
    checkpoint_config = config["logging"]["checkpoint"]
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_config["dirpath"],
        filename=checkpoint_config["filename"],
        save_top_k=checkpoint_config.get("save_top_k", 3),
        mode=checkpoint_config.get("mode", "min"),
        monitor=checkpoint_config.get("monitor", "epoch/val_loss_total"),
        save_last=checkpoint_config.get("save_last", True),
        every_n_epochs=checkpoint_config.get("every_n_epochs", 10),
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    early_stop_config = config["logging"]["early_stopping"]
    if early_stop_config.get("enabled", True):
        early_stop_callback = EarlyStopping(
            monitor=early_stop_config.get("monitor", "epoch/val_loss_total"),
            patience=early_stop_config.get("patience", 50),
            mode=early_stop_config.get("mode", "min"),
            min_delta=early_stop_config.get("min_delta", 0.001),
        )
        callbacks.append(early_stop_callback)

    logger.info(f"✅ Callbacks configurados: {len(callbacks)} callbacks")
    return callbacks


def create_loggers(config: Dict[str, Any]) -> list:
    """Cria loggers para o treinamento."""
    loggers = []

    # TensorBoard logger
    tb_config = config["logging"]["tensorboard"]
    tb_logger = TensorBoardLogger(
        save_dir=tb_config["save_dir"],
        name=tb_config["name"],
        version=tb_config.get("version"),
    )
    loggers.append(tb_logger)

    # Weights & Biases logger (opcional)
    wandb_config = config["logging"].get("wandb", {})
    if wandb_config.get("enabled", False):
        try:
            wandb_logger = WandbLogger(
                project=wandb_config["project"],
                name=wandb_config["name"],
                tags=wandb_config.get("tags", []),
            )
            loggers.append(wandb_logger)
            logger.info("✅ Weights & Biases configurado")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao configurar W&B: {e}")

    logger.info(f"✅ Loggers configurados: {len(loggers)} loggers")
    return loggers


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Treinamento VITS2 com Monitoramento LLM"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Caminho para arquivo de configuração YAML",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Caminho para checkpoint para resumir treinamento",
    )
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="Desabilitar monitoramento LLM",
    )

    args = parser.parse_args()

    # Criar diretórios necessários
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    logger.info("🚀 Iniciando treinamento VITS2 com monitoramento LLM")
    logger.info(f"📁 Configuração: {args.config}")

    try:
        # Carregar configuração
        config = load_config(args.config)

        # Configurar monitoramento LLM
        llm_monitor_config = None
        if not args.disable_llm:
            llm_monitor_config = create_llm_monitor_config(config)

        # Criar modelo
        model = create_model(config)

        # Criar trainer
        trainer_module = VITS2Trainer(
            model=model,
            config=config,  # Passar configuração completa
            llm_monitor_config=llm_monitor_config,
        )

        # Criar data loaders
        train_loader, val_loader = create_data_loaders(config)

        # Configurar gerador de amostras baseado no dataset
        trainer_module.setup_audio_sampler_for_dataset(train_loader.dataset)

        # Criar callbacks
        callbacks = create_callbacks(config)

        # Criar loggers
        loggers = create_loggers(config)

        # Configurar Lightning Trainer
        hardware_config = config["hardware"]

        # Configurações de teste (se existirem)
        test_config = config.get("test", {})

        trainer = L.Trainer(
            max_epochs=config["training"]["max_epochs"],
            accelerator=hardware_config.get("accelerator", "gpu"),
            devices=hardware_config.get("devices", 1),
            precision=hardware_config.get("precision", "16-mixed"),
            strategy=hardware_config.get("strategy", "auto"),
            accumulate_grad_batches=config["training"].get(
                "accumulate_grad_batches", 1
            ),
            gradient_clip_val=config["training"].get("gradient_clip_val", 1.0),
            val_check_interval=config["validation"].get(
                "val_check_interval", 1.0
            ),
            limit_val_batches=config["validation"].get(
                "limit_val_batches", 1.0
            ),
            limit_train_batches=test_config.get(
                "limit_train_batches", 1.0
            ),  # 🚀 TESTE
            num_sanity_val_steps=config["validation"].get(
                "num_sanity_val_steps", 2
            ),
            fast_dev_run=test_config.get("fast_dev_run", False),  # 🚀 TESTE
            overfit_batches=test_config.get("overfit_batches", 0),  # 🚀 TESTE
            callbacks=callbacks,
            logger=loggers,
            benchmark=hardware_config.get("benchmark", True),
            deterministic=hardware_config.get("deterministic", False),
            sync_batchnorm=hardware_config.get("sync_batchnorm", True),
        )

        # Iniciar treinamento
        logger.info("🎯 Iniciando treinamento...")

        if llm_monitor_config:
            logger.info(
                "🤖 Monitoramento LLM ATIVO - Sistema pioneiro em funcionamento!"
            )

        if args.resume:
            logger.info(
                f"📁 Resumindo treinamento do checkpoint: {args.resume}"
            )
            trainer.fit(
                trainer_module, train_loader, val_loader, ckpt_path=args.resume
            )
        else:
            trainer.fit(trainer_module, train_loader, val_loader)

        # Estatísticas finais
        if llm_monitor_config and trainer_module.llm_monitor:
            stats = trainer_module.get_llm_statistics()
            if stats:
                logger.info("📊 Estatísticas do Monitoramento LLM:")
                logger.info(
                    f"   • Total de análises: {stats['total_analyses']}"
                )
                logger.info(
                    f"   • Sugestões aplicadas: {stats['suggestions_applied']}"
                )
                logger.info(
                    f"   • Sugestões rejeitadas: {stats['suggestions_rejected']}"
                )
                logger.info(
                    f"   • Mudanças críticas: {stats.get('critical_changes', 0)}"
                )

        logger.info("🎉 Treinamento concluído com sucesso!")

    except Exception as e:
        logger.error(f"❌ Erro durante o treinamento: {e}")
        raise


if __name__ == "__main__":
    main()
