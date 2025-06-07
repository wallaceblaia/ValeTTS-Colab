"""API models placeholder."""

from pydantic import BaseModel


class TTSRequest(BaseModel):
    """TTS request model."""
    text: str


class TTSResponse(BaseModel):
    """TTS response model."""
    audio: bytes
