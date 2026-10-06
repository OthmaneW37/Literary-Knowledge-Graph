from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import VisualObservation


class VisionProvider(Protocol):
    def analyze(self, image_path: str | Path, question: str = "") -> VisualObservation: ...
