#!/usr/bin/env python3
"""
Script de inferência para ValeTTS.

Uso:
    # Síntese básica
    python scripts/inference.py --text "Hello world" --output output.wav
    
    # Com clonagem de voz
    python scripts/inference.py --text "Hello world" \
        --reference_audio speaker.wav --output cloned.wav
        
    # Multilíngue
    python scripts/inference.py --text "Olá mundo" \
        --language pt --output portuguese.wav
        
    # Com controles prosódicos
    python scripts/inference.py --text "Hello world" \
        --pitch_scale 1.2 --speed_scale 0.8 --output prosody.wav
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import soundfile as sf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from valetts import TTSSynthesizer, VoiceCloner
from valetts.utils.audio import AudioProcessor


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ValeTTS Inference Script")
    
    # Required arguments
    parser.add_argument("--text", type=str, required=True,
                       help="Text to synthesize")
    parser.add_argument("--output", type=str, required=True,
                       help="Output audio file path")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="models/vits2_base.ckpt",
                       help="Path to trained model checkpoint")
    parser.add_argument("--config_path", type=str, default="configs/models/vits2_base.yaml",
                       help="Path to model configuration")
    
    # Voice cloning arguments
    parser.add_argument("--reference_audio", type=str, default=None,
                       help="Reference audio for voice cloning")
    parser.add_argument("--speaker_id", type=str, default=None,
                       help="Speaker ID for multi-speaker synthesis")
    
    # Language arguments
    parser.add_argument("--language", type=str, default="en",
                       choices=["en", "pt", "es", "fr", "de", "it", "ja", "zh"],
                       help="Language code for synthesis")
    
    # Prosody control arguments
    parser.add_argument("--pitch_scale", type=float, default=1.0,
                       help="Pitch scale factor (0.5-2.0)")
    parser.add_argument("--speed_scale", type=float, default=1.0,
                       help="Speed scale factor (0.5-2.0)")
    parser.add_argument("--energy_scale", type=float, default=1.0,
                       help="Energy scale factor (0.5-2.0)")
    
    # Audio arguments
    parser.add_argument("--sample_rate", type=int, default=22050,
                       help="Output sample rate")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cpu, cuda, auto)")
    
    return parser.parse_args()


def setup_device(device: str) -> torch.device:
    """Setup computation device."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch_device = torch.device(device)
    print(f"Using device: {torch_device}")
    
    if torch_device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return torch_device


def main():
    """Main inference function."""
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Setup prosody controls
    prosody_controls = {
        "pitch_scale": args.pitch_scale,
        "speed_scale": args.speed_scale, 
        "energy_scale": args.energy_scale,
    }
    
    # Initialize synthesizer or voice cloner
    if args.reference_audio:
        print("Initializing voice cloner...")
        cloner = VoiceCloner(
            model_path=args.model_path,
            config_path=args.config_path,
            device=device
        )
        
        print(f"Cloning voice from: {args.reference_audio}")
        audio = cloner.clone_voice(
            text=args.text,
            reference_audio=args.reference_audio,
            language=args.language,
            prosody_controls=prosody_controls
        )
        
    else:
        print("Initializing TTS synthesizer...")
        synthesizer = TTSSynthesizer(
            model_path=args.model_path,
            config_path=args.config_path,
            device=device
        )
        
        print(f"Synthesizing text: '{args.text}'")
        audio = synthesizer.synthesize(
            text=args.text,
            speaker_id=args.speaker_id,
            language=args.language,
            prosody_controls=prosody_controls
        )
    
    # Process audio
    audio_processor = AudioProcessor(sample_rate=args.sample_rate)
    audio = audio_processor.postprocess(audio)
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    sf.write(output_path, audio.cpu().numpy(), args.sample_rate)
    print(f"Audio saved to: {output_path}")
    
    # Print statistics
    duration = len(audio) / args.sample_rate
    print(f"Generated audio duration: {duration:.2f} seconds")
    print(f"Audio shape: {audio.shape}")


if __name__ == "__main__":
    main() 