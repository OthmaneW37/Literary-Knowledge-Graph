from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    version: str
    def rerank(self, query: str, passages: list, top_k: int) -> list: ...
