from __future__ import annotations

from .cross_encoder import CrossEncoderReranker


class TrainedReranker(CrossEncoderReranker):
    """Loads a locally trained checkpoint; model loading stays lazy."""

    def __init__(self, model_path: str, model=None) -> None:
        super().__init__(model_path, model=model)
