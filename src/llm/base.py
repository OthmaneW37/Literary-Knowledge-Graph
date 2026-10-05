from __future__ import annotations

from typing import Any, Protocol


class LLMProvider(Protocol):
    """Application boundary for local chat generation and embedding requests."""

    name: str

    def chat(self, **kwargs: Any) -> Any: ...

    def embed(self, **kwargs: Any) -> Any: ...
