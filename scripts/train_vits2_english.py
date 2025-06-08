#!/usr/bin/env python3
"""
Script de treinamento VITS2 específico para inglês.

Este script é otimizado para datasets em inglês com:
- Processamento de texto específico para inglês
- Phonemes G2P corretos
- Configurações otimizadas para A100/V100/T4

Uso:
    python scripts/train_vits2_english.py --config configs/training/vits2_english_a100_optimized.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from valetts.data.loaders.english_tts import create_english_dataloader
from valetts.training.trainers.vits2 import VITS2Trainer
from valetts.models.vits2.model import VITS2
from valetts.models.vits2.config import VITS2Config
from valetts.training.callbacks.hybrid_checkpoint import create_hybrid_checkpoint_callbacks

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/training.log")
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Carrega configuração YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> VITS2:
    """Cria modelo VITS2 com configuração."""
    model_config = config['model']
    
    # Verificar se configurações são compatíveis
    speaker_dim = model_config['speaker_embedding_dim']
    decoder_dim = model_config['decoder_hidden_dim']
    
    if speaker_dim != decoder_dim:
        logger.warning(f"⚠️ Dimensões incompatíveis detectadas!")
        logger.warning(f"   Speaker embedding: {speaker_dim}")
        logger.warning(f"   Decoder hidden: {decoder_dim}")
        logger.warning(f"   Ajustando decoder para {speaker_dim}")
        model_config['decoder_hidden_dim'] = speaker_dim
    
    # Criar configuração VITS2
    vits2_config = VITS2Config(**model_config)
    
    # Criar modelo com configuração
    model = VITS2(vits2_config)
    logger.info(f"✅ Modelo VITS2 criado com {sum(p.numel() for p in model.parameters()):,} parâmetros")
    
    return model


def create_trainer_module(model: VITS2, config: dict) -> VITS2Trainer:
    """Cria módulo de treinamento Lightning."""
    training_config = config['training']
    
    trainer_module = VITS2Trainer(
        model=model,
        learning_rate=training_config['learning_rate'],
        **training_config
    )
    
    logger.info("✅ Módulo de treinamento criado")
    return trainer_module


def setup_callbacks(config: dict):
    """Configura callbacks do Lightning."""
    callbacks = []
    
    # Sistema híbrido de checkpoints
    hybrid_callbacks = create_hybrid_checkpoint_callbacks(config['logging'])
    callbacks.extend(hybrid_callbacks)
    
    # Early stopping callback
    if config['logging']['early_stopping'].get('enabled', True):
        early_stop_config = config['logging']['early_stopping']
        early_stop_callback = EarlyStopping(
            monitor=early_stop_config['monitor'],
            mode=early_stop_config['mode'],
            patience=early_stop_config['patience'],
            min_delta=early_stop_config['min_delta'],
        )
        callbacks.append(early_stop_callback)
    
    logger.info(f"✅ Configurados {len(callbacks)} callbacks")
    logger.info(f"   📦 Checkpoints híbridos: 2 recentes + backup a cada época")
    return callbacks


def setup_logger(config: dict):
    """Configura logger do TensorBoard."""
    tensorboard_config = config['logging']['tensorboard']
    
    tb_logger = TensorBoardLogger(
        save_dir=tensorboard_config['save_dir'],
        name=tensorboard_config['name'],
    )
    
    logger.info(f"📊 TensorBoard configurado: {tensorboard_config['save_dir']}")
    return tb_logger


def main():
    parser = argparse.ArgumentParser(description="Treinamento VITS2 para inglês")
    parser.add_argument("--config", type=str, required=True, help="Caminho para arquivo de configuração")
    parser.add_argument("--resume", type=str, help="Caminho para checkpoint para resumir treinamento")
    parser.add_argument("--disable-llm", action="store_true", help="Desabilitar monitor LLM")
    args = parser.parse_args()
    
    # Carregar configuração
    logger.info(f"📋 Carregando configuração: {args.config}")
    config = load_config(args.config)
    
    # Desabilitar LLM se solicitado
    if args.disable_llm:
        config['llm_monitor']['enabled'] = False
        logger.info("🤖 Monitor LLM desabilitado")
    
    # Verificar se é dataset em inglês
    expected_locale = config['dataset_config']['expected_locale']
    if expected_locale != 'en':
        logger.warning(f"⚠️ Configuração não é para inglês (locale={expected_locale})")
    
    # Criar diretórios necessários
    os.makedirs(config['logging']['checkpoint']['dirpath'], exist_ok=True)
    os.makedirs(config['logging']['tensorboard']['save_dir'], exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Criar diretório de backup se habilitado
    if config['logging'].get('checkpoint_backup', {}).get('enabled', False):
        backup_dir = config['logging']['checkpoint_backup']['dirpath']
        os.makedirs(backup_dir, exist_ok=True)
        logger.info(f"🗄️ Diretório de backup criado: {backup_dir}")
    
    logger.info("🚀 Iniciando treinamento VITS2 para inglês")
    logger.info(f"   📊 Configuração: {Path(args.config).name}")
    logger.info(f"   🌍 Idioma: inglês (locale={expected_locale})")
    logger.info(f"   📦 Batch size: {config['training']['batch_size']}")
    logger.info(f"   🔄 Épocas: {config['training']['max_epochs']}")
    
    # Detectar GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        logger.warning("⚠️ CUDA não disponível, usando CPU")
    
    # Criar dataloaders
    logger.info("📊 Criando dataloaders...")
    
    train_dataloader = create_english_dataloader(
        data_dir=config['data']['data_dir'],
        metadata_file=config['data']['metadata_file'],
        audio_config={
            'sample_rate': config['data']['sample_rate'],
            'n_mels': config['data']['n_mels'],
            'n_fft': config['data']['n_fft'],
            'hop_length': config['data']['hop_length'],
            'win_length': config['data']['win_length'],
        },
        text_config=config['data']['text_processor'],
        split='train',
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        expected_locale=config['dataset_config']['expected_locale'],
        locale_column=config['dataset_config']['locale_column'],
        audio_column=config['dataset_config']['audio_column'],
        text_column=config['dataset_config']['text_column'],
        speaker_column=config['dataset_config']['speaker_column'],
        max_samples=config['data'].get('max_samples_debug'),
    )
    
    val_dataloader = create_english_dataloader(
        data_dir=config['data']['data_dir'],
        metadata_file=config['data']['metadata_file'],
        audio_config={
            'sample_rate': config['data']['sample_rate'],
            'n_mels': config['data']['n_mels'],
            'n_fft': config['data']['n_fft'],
            'hop_length': config['data']['hop_length'],
            'win_length': config['data']['win_length'],
        },
        text_config=config['data']['text_processor'],
        split='val',
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        expected_locale=config['dataset_config']['expected_locale'],
        locale_column=config['dataset_config']['locale_column'],
        audio_column=config['dataset_config']['audio_column'],
        text_column=config['dataset_config']['text_column'],
        speaker_column=config['dataset_config']['speaker_column'],
        max_samples=config['data'].get('max_samples_debug'),
    )
    
    # Atualizar configuração do modelo com número real de speakers
    n_speakers = train_dataloader.dataset.n_speakers
    config['model']['n_speakers'] = n_speakers
    logger.info(f"🎤 Speakers detectados: {n_speakers}")
    
    # Criar modelo
    logger.info("🧠 Criando modelo...")
    model = create_model(config)
    
    # Criar módulo de treinamento
    logger.info("⚡ Configurando treinamento...")
    trainer_module = create_trainer_module(model, config)
    
    # Configurar callbacks e logger
    callbacks = setup_callbacks(config)
    tb_logger = setup_logger(config)
    
    # Configurar trainer do Lightning
    hardware_config = config['hardware']
    trainer = Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=hardware_config['accelerator'],
        devices=hardware_config['devices'],
        precision=hardware_config['precision'],
        strategy=hardware_config['strategy'],
        callbacks=callbacks,
        logger=tb_logger,
        val_check_interval=config['validation']['val_check_interval'],
        limit_val_batches=config['validation']['limit_val_batches'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        benchmark=hardware_config.get('benchmark', False),
        deterministic=hardware_config.get('deterministic', False),
    )
    
    logger.info("🏁 Iniciando treinamento...")
    
    # Treinar
    try:
        trainer.fit(
            trainer_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=args.resume
        )
        
        logger.info("🎉 Treinamento concluído com sucesso!")
        
        # Salvar modelo final
        final_model_path = Path(config['logging']['checkpoint']['dirpath']) / "final_model.ckpt"
        trainer.save_checkpoint(str(final_model_path))
        logger.info(f"💾 Modelo final salvo: {final_model_path}")
        
    except Exception as e:
        logger.error(f"❌ Erro durante treinamento: {e}")
        raise
    
    except KeyboardInterrupt:
        logger.info("⏸️ Treinamento interrompido pelo usuário")


if __name__ == "__main__":
    main()