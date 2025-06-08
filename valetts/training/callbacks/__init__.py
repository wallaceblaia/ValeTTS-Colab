"""
Training callbacks for ValeTTS.

Callbacks personalizados para treinamento:
- Hybrid checkpoint system
- Audio sample generation
- LLM monitoring integration
"""

from .hybrid_checkpoint import HybridCheckpointCallback, create_hybrid_checkpoint_callbacks

__all__ = [
    'HybridCheckpointCallback',
    'create_hybrid_checkpoint_callbacks',
]