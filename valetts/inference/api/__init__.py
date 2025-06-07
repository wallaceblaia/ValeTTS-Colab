"""
APIs de produção para ValeTTS.
"""

from valetts.inference.api.endpoints import router
from valetts.inference.api.models import TTSRequest, TTSResponse
from valetts.inference.api.server import TTSServer

__all__ = [
    "TTSServer",
    "router",
    "TTSRequest",
    "TTSResponse",
]
