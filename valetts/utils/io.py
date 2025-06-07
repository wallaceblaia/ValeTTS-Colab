"""
Utilitários de entrada/saída para o ValeTTS.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Carrega um arquivo de configuração YAML.

    Args:
        config_path: Caminho para o arquivo de configuração

    Returns:
        Dicionário com as configurações
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Salva um dicionário de configuração em arquivo YAML.

    Args:
        config: Dicionário com as configurações
        config_path: Caminho para salvar o arquivo
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def load_model(
    model_path: Union[str, Path], map_location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Carrega um modelo salvo.

    Args:
        model_path: Caminho para o arquivo do modelo
        map_location: Dispositivo onde carregar o modelo

    Returns:
        Dicionário com os dados do modelo
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    return torch.load(model_path, map_location=map_location)


def save_model(
    model_data: Dict[str, Any], model_path: Union[str, Path]
) -> None:
    """
    Salva dados do modelo.

    Args:
        model_data: Dados do modelo para salvar
        model_path: Caminho onde salvar o modelo
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model_data, model_path)


def load_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Carrega um arquivo JSON.

    Args:
        json_path: Caminho para o arquivo JSON

    Returns:
        Dicionário com os dados JSON
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def save_json(data: Dict[str, Any], json_path: Union[str, Path]) -> None:
    """
    Salva dados em arquivo JSON.

    Args:
        data: Dados para salvar
        json_path: Caminho onde salvar o arquivo JSON
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    Garante que um diretório existe, criando-o se necessário.

    Args:
        dir_path: Caminho do diretório

    Returns:
        Path do diretório
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """
    Encontra o checkpoint mais recente em um diretório.

    Args:
        checkpoint_dir: Diretório de checkpoints

    Returns:
        Caminho para o checkpoint mais recente ou None se não encontrado
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("*.ckpt"))

    if not checkpoints:
        return None

    # Ordenar por tempo de modificação (mais recente primeiro)
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return checkpoints[0]
