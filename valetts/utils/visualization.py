"""
Utilitários de visualização para o ValeTTS.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_mel_spectrogram(
    mel: torch.Tensor,
    title: str = "Mel Spectrogram",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> None:
    """
    Plota um espectrograma mel.

    Args:
        mel: Tensor do espectrograma mel (formato: [mel_bins, frames])
        title: Título do gráfico
        figsize: Tamanho da figura
        save_path: Caminho para salvar o gráfico (opcional)
    """
    # Converter tensor para numpy se necessário
    if isinstance(mel, torch.Tensor):
        mel_np = mel.detach().cpu().numpy()
    else:
        mel_np = mel

    plt.figure(figsize=figsize)
    plt.imshow(mel_np, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Frames")
    plt.ylabel("Mel Frequency Bins")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_attention(
    attention: torch.Tensor,
    title: str = "Attention",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> None:
    """
    Plota uma matriz de atenção.

    Args:
        attention: Tensor da atenção (formato: [encoder_len, decoder_len])
        title: Título do gráfico
        figsize: Tamanho da figura
        save_path: Caminho para salvar o gráfico (opcional)
    """
    # Converter tensor para numpy se necessário
    if isinstance(attention, torch.Tensor):
        attention_np = attention.detach().cpu().numpy()
    else:
        attention_np = attention

    plt.figure(figsize=figsize)
    plt.imshow(
        attention_np, aspect="auto", origin="lower", interpolation="none"
    )
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Decoder Steps")
    plt.ylabel("Encoder Steps")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_training_metrics(
    train_losses: list,
    val_losses: list,
    title: str = "Training Metrics",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plota métricas de treinamento.

    Args:
        train_losses: Lista de perdas de treinamento
        val_losses: Lista de perdas de validação
        title: Título do gráfico
        figsize: Tamanho da figura
        save_path: Caminho para salvar o gráfico (opcional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot de perdas
    ax1.plot(train_losses, label="Train Loss", color="blue")
    ax1.plot(val_losses, label="Val Loss", color="red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot de learning rate (se disponível)
    ax2.plot(train_losses, label="Train Loss", color="blue")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Loss (Log Scale)")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_audio_waveform(
    audio: torch.Tensor,
    sample_rate: int = 22050,
    title: str = "Audio Waveform",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> None:
    """
    Plota forma de onda de áudio.

    Args:
        audio: Tensor de áudio
        sample_rate: Taxa de amostragem
        title: Título do gráfico
        figsize: Tamanho da figura
        save_path: Caminho para salvar o gráfico (opcional)
    """
    # Converter tensor para numpy se necessário
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().numpy()
    else:
        audio_np = audio

    # Criar eixo temporal
    time_axis = np.linspace(0, len(audio_np) / sample_rate, len(audio_np))

    plt.figure(figsize=figsize)
    plt.plot(time_axis, audio_np, linewidth=0.5)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
