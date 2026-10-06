from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .spoiler_policy import SpoilerPolicy


@dataclass(frozen=True)
class VisualRecord:
    visual_id: str
    work_id: str
    chapter: int | None
    page: int | None
    local_path: str
    caption: str = ""
    surrounding_text: str = ""
    ocr_text: str = ""
    visual_type: str = "other"
    score: float = 0.0


class VisualEmbeddingProvider(Protocol):
    def embed_image(self, path: str | Path) -> np.ndarray: ...
    def embed_text(self, text: str) -> np.ndarray: ...


class TransformersClipProvider:
    """Image CLIP encoder aligned with its multilingual text encoder."""

    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
                 image_model_name: str = "clip-ViT-B-32") -> None:
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.image_model_name = image_model_name
        self.cache_version = f"image-{image_model_name}__text-{model_name}"
        self.image_model = SentenceTransformer(image_model_name)
        self.text_model = SentenceTransformer(model_name)

    def embed_image(self, path: str | Path) -> np.ndarray:
        from PIL import Image
        return np.asarray(self.image_model.encode(Image.open(path).convert("RGB"), convert_to_numpy=True), dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        return np.asarray(self.text_model.encode(text, convert_to_numpy=True), dtype=np.float32)


class LocalVisualIndex:
    """Filesystem metadata plus an optional local joint-embedding matrix."""

    def __init__(self, metadata_path: str | Path, provider: VisualEmbeddingProvider | None = None) -> None:
        self.metadata_path = Path(metadata_path)
        self.provider = provider
        self.records = self._load()
        self.cache_path: Path | None = None
        if self.provider is not None:
            model_name = str(getattr(self.provider, "cache_version", getattr(self.provider, "model_name", type(self.provider).__name__)))
            safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name)
            root = self.metadata_path.parent.parent if self.metadata_path.is_dir() or self.metadata_path.parent.name == "processed" else self.metadata_path.parent
            self.cache_path = root / "library" / "visual_embeddings" / f"{safe}.npz"

    def _load(self) -> list[VisualRecord]:
        paths = sorted(self.metadata_path.glob("*.visuals.json")) if self.metadata_path.is_dir() else [self.metadata_path]
        records = []
        for path in paths:
            try:
                values = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for item in values:
                if not isinstance(item, dict) or not item.get("local_path"):
                    continue
                allowed = {key: item[key] for key in ("visual_id", "work_id", "chapter", "page", "local_path", "caption", "surrounding_text", "ocr_text") if key in item}
                allowed["visual_type"] = item.get("visual_type", item.get("type", "other"))
                records.append(VisualRecord(**allowed))
        return records

    def search(self, query: str, work_ids: list[str], top_k: int = 5, max_chapter: int | None = None,
               image_path: str | Path | None = None) -> list[VisualRecord]:
        allowed = [r for r in self.records if r.work_id in set(work_ids) and SpoilerPolicy.allows(r.chapter, max_chapter)]
        if not allowed or self.provider is None:
            return []
        query_vector = self.provider.embed_image(image_path) if image_path else self.provider.embed_text(query)
        fingerprints = []
        for item in allowed:
            content = Path(item.local_path).read_bytes()
            fingerprints.append(hashlib.sha256(content).hexdigest())
        cached = self._load_vectors()
        vectors_list = []
        changed = False
        for item, fingerprint in zip(allowed, fingerprints):
            vector = cached.get(fingerprint)
            if vector is None:
                vector = np.asarray(self.provider.embed_image(item.local_path), dtype=np.float32)
                cached[fingerprint] = vector
                changed = True
            vectors_list.append(vector)
        if changed:
            self._save_vectors(cached)
        vectors = np.stack(vectors_list)
        query_vector = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
        vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        scores = vectors @ query_vector
        return [VisualRecord(**{**item.__dict__, "score": float(scores[i])}) for i, item in sorted(enumerate(allowed), key=lambda pair: (-float(scores[pair[0]]), pair[0]))[:top_k]]

    def _load_vectors(self) -> dict[str, np.ndarray]:
        if self.cache_path is None or not self.cache_path.is_file():
            return {}
        try:
            with np.load(self.cache_path, allow_pickle=False) as payload:
                keys = payload["fingerprints"].astype(str).tolist()
                vectors = np.asarray(payload["vectors"], dtype=np.float32)
            return {key: vector for key, vector in zip(keys, vectors)} if len(keys) == len(vectors) else {}
        except (OSError, ValueError, KeyError):
            return {}

    def _save_vectors(self, vectors: dict[str, np.ndarray]) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        items = list(vectors.items())
        temporary = self.cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, fingerprints=np.asarray([key for key, _ in items]), vectors=np.stack([value for _, value in items]))
        temporary.replace(self.cache_path)


class MultimodalRetriever:
    """Fuse separately ranked text and visual results with reciprocal rank fusion."""

    @staticmethod
    def fuse(text_results, visual_results, limit: int = 8, rrf_k: int = 60):
        scores: dict[tuple[str, str], float] = {}
        values: dict[tuple[str, str], object] = {}
        for results in (text_results, visual_results):
            for rank, item in enumerate(results, start=1):
                key = ("visual" if isinstance(item, VisualRecord) else "text", getattr(item, "visual_id", getattr(item, "chunk_id", "")))
                scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + rank)
                values[key] = item
        return [values[key] for key in sorted(values, key=lambda key: (-scores[key], key))[:limit]]
