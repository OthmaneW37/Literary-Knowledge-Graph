from __future__ import annotations

import os
from pathlib import Path

from .models import VisualObservation
from .ollama_vlm import OllamaVisionProvider
from .provider import VisionProvider


class VisionAnalyzer:
    """Lazy VLM facade. Text-only operation never imports or calls a VLM."""

    def __init__(self, provider: VisionProvider | None = None) -> None:
        self.provider = provider

    @property
    def enabled(self) -> bool:
        return os.getenv("VLM_PROVIDER", "ollama").casefold() not in {"", "none", "disabled"}

    def analyze(self, image_path: str | Path, question: str = "") -> VisualObservation | None:
        if not self.enabled:
            return None
        provider = self.provider or OllamaVisionProvider()
        return provider.analyze(image_path, question)
