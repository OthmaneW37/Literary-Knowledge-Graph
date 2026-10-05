# src/retrieval/lexical_retriever.py

from __future__ import annotations

from pathlib import Path

from rag.local_index import LocalLiteraryIndex
from rag.models import Passage


class LexicalRetriever:
    """
    Adaptateur autour de LocalLiteraryIndex.

    Cette classe permet au HybridRetriever d'utiliser le moteur BM25
    sans dépendre directement de son implémentation.
    """

    def __init__(
        self,
        manifest_path: str | Path = "data/annotations/work_manifest.json",
        processed_dir: str | Path = "data/processed",
    ) -> None:

        self.index = LocalLiteraryIndex(
            manifest_path,
            processed_dir,
        )

    def retrieve(
        self,
        query: str,
        work_ids: list[str] | None = None,
        top_k: int = 6,
        max_chapter: int | None = None,
    ) -> list[Passage]:

        return self.index.search(
            query,
            work_ids=work_ids,
            top_k=top_k,
            max_chapter=max_chapter,
        )
