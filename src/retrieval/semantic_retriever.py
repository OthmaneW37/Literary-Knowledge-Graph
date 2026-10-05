from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import ollama

from rag.local_index import LocalLiteraryIndex
from rag.models import Passage


LOGGER = logging.getLogger(__name__)
QUERY_INSTRUCTION = (
    "Instruct: Given a literary question, retrieve passages that contain the evidence "
    "needed to answer it.\nQuery: "
)


class SemanticRetriever:
    """Local multilingual vector search backed by Ollama embeddings.

    Passage vectors are persisted on disk and only new or changed chunks are
    embedded. The query vector is the only embedding computed during a normal
    chat request.
    """

    def __init__(
        self,
        index: LocalLiteraryIndex,
        model: str = "qwen3-embedding:0.6b",
        cache_dir: str | Path = "data/library/embeddings",
        batch_size: int = 16,
        keep_alive: str = "15m",
        embedding_provider: object | None = None,
    ) -> None:
        self.index = index
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.batch_size = max(1, batch_size)
        self.keep_alive = keep_alive
        self.embedding_provider = embedding_provider
        safe_model_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", model)
        self.cache_path = self.cache_dir / f"{safe_model_name}.npz"
        self._vectors: np.ndarray | None = None
        self._ready = False

    @staticmethod
    def _document_text(passage: Passage) -> str:
        return (
            f"Title: {passage.work_title}\n"
            f"Chapter: {passage.chapter}\n"
            f"Passage: {passage.text}"
        )

    @classmethod
    def _fingerprint(cls, passage: Passage) -> str:
        return hashlib.sha256(cls._document_text(passage).encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, np.finfo(np.float32).eps)

    @staticmethod
    def _response_vectors(response: object) -> np.ndarray:
        raw = getattr(response, "embeddings", None)
        if raw is None and isinstance(response, dict):
            raw = response.get("embeddings")
        vectors = np.asarray(raw, dtype=np.float32)
        if vectors.ndim != 2 or not vectors.size:
            raise ValueError("Ollama returned no usable embedding vectors")
        return vectors

    def _embed(self, texts: list[str]) -> np.ndarray:
        embed = (
            getattr(self.embedding_provider, "embed")
            if self.embedding_provider is not None
            else ollama.embed
        )
        response = embed(
            model=self.model,
            input=texts,
            truncate=True,
            keep_alive=self.keep_alive,
        )
        return self._normalize(self._response_vectors(response))

    def _load_cache(self) -> dict[tuple[str, str], np.ndarray]:
        if not self.cache_path.exists():
            return {}
        try:
            with np.load(self.cache_path, allow_pickle=False) as payload:
                if str(payload["model"].item()) != self.model:
                    return {}
                ids = payload["ids"].astype(str).tolist()
                fingerprints = payload["fingerprints"].astype(str).tolist()
                vectors = np.asarray(payload["vectors"], dtype=np.float32)
            if len(ids) != len(fingerprints) or len(ids) != len(vectors):
                return {}
            return {
                (chunk_id, fingerprint): vector
                for chunk_id, fingerprint, vector in zip(ids, fingerprints, vectors)
            }
        except (OSError, ValueError, KeyError):
            LOGGER.warning("Ignoring an invalid semantic cache at %s", self.cache_path)
            return {}

    def _save_cache(self, fingerprints: list[str], vectors: np.ndarray) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = self.cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary_path,
            model=np.asarray(self.model),
            ids=np.asarray([passage.chunk_id for passage in self.index.passages]),
            fingerprints=np.asarray(fingerprints),
            vectors=vectors.astype(np.float32),
        )
        temporary_path.replace(self.cache_path)

    def prepare(self, force: bool = False, progress=None) -> int:
        """Build or refresh the vector cache and return its passage count."""
        if not force and self._ready and self._vectors is not None:
            return len(self._vectors)

        fingerprints = [self._fingerprint(passage) for passage in self.index.passages]
        cached = {} if force else self._load_cache()
        vectors_by_position: list[np.ndarray | None] = [None] * len(self.index.passages)
        missing_positions: list[int] = []
        for position, (passage, fingerprint) in enumerate(
            zip(self.index.passages, fingerprints)
        ):
            vector = cached.get((passage.chunk_id, fingerprint))
            if vector is None:
                missing_positions.append(position)
            else:
                vectors_by_position[position] = vector

        for start in range(0, len(missing_positions), self.batch_size):
            batch_positions = missing_positions[start : start + self.batch_size]
            batch_vectors = self._embed(
                [self._document_text(self.index.passages[position]) for position in batch_positions]
            )
            if len(batch_vectors) != len(batch_positions):
                raise ValueError("Ollama returned an unexpected embedding batch size")
            for position, vector in zip(batch_positions, batch_vectors):
                vectors_by_position[position] = vector
            if progress:
                progress(min(start + len(batch_positions), len(missing_positions)), len(missing_positions))

        if not vectors_by_position:
            raise ValueError("The literary index contains no passages")
        if any(vector is None for vector in vectors_by_position):
            raise ValueError("The semantic cache could not be completed")

        dimensions = {len(vector) for vector in vectors_by_position if vector is not None}
        if len(dimensions) != 1:
            # A model update can change dimensions while retaining the same
            # Ollama tag. Rebuild the whole cache in that rare case.
            all_vectors: list[np.ndarray] = []
            for start in range(0, len(self.index.passages), self.batch_size):
                batch = self.index.passages[start : start + self.batch_size]
                all_vectors.extend(self._embed([self._document_text(item) for item in batch]))
            vectors_by_position = all_vectors

        self._vectors = self._normalize(np.stack(vectors_by_position).astype(np.float32))
        if missing_positions or not self.cache_path.exists():
            self._save_cache(fingerprints, self._vectors)
        self._ready = True
        return len(self._vectors)

    def retrieve(
        self,
        query: str,
        work_ids: list[str] | None = None,
        top_k: int = 6,
        max_chapter: int | None = None,
    ) -> list[Passage]:
        self.prepare()
        if self._vectors is None or top_k < 1:
            return []

        query_vector = self._embed([f"{QUERY_INSTRUCTION}{query}"])[0]
        if self._vectors.shape[1] != query_vector.shape[0]:
            # The local tag may have been updated to an embedding with a new
            # dimension. Refresh the cache automatically instead of leaving
            # semantic search permanently disabled.
            self.prepare(force=True)
            query_vector = self._embed([f"{QUERY_INSTRUCTION}{query}"])[0]
        scores = self._vectors @ query_vector
        selected_works = set(self.index.works if work_ids is None else work_ids)
        eligible = [
            position
            for position, passage in enumerate(self.index.passages)
            if passage.work_id in selected_works
            and (
                max_chapter is None
                or (str(passage.chapter).isdigit() and int(passage.chapter) <= max_chapter)
            )
        ]
        eligible.sort(key=lambda position: (-float(scores[position]), position))
        return [
            replace(self.index.passages[position], score=round(float(scores[position]), 6))
            for position in eligible[:top_k]
        ]
