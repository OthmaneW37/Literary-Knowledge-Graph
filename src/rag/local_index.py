from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .models import Passage, Work


TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", flags=re.UNICODE)

# Frequent French and English words carry little retrieval value. Keeping the
# list deliberately small preserves literary names and meaningful expressions.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "de", "des", "du", "en", "est", "et", "il", "elle", "ils", "elles",
    "for", "from", "he", "her", "his", "how", "in", "is", "it", "its",
    "la", "le", "les", "leur", "lui", "mais", "of", "on", "or", "ou",
    "par", "pas", "pour", "que", "qui", "sa", "se", "ses", "she",
    "son", "sur", "the", "their", "them", "they", "this", "to", "un",
    "une", "was", "were", "what", "when", "where", "which", "who",
    "why", "with", "you", "à", "ça",
}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    return "".join(ch for ch in value if not unicodedata.combining(ch))


def tokenize(value: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(normalize_text(value))
    return [token.replace("’", "'") for token in tokens if len(token) > 1 and token not in STOP_WORDS]


class LocalLiteraryIndex:
    """Small in-memory BM25 index over the locally processed novels.

    The initial corpus is intentionally small, so this requires no database and
    makes the MVP usable immediately. It can later be complemented by vector
    embeddings without changing the public search API.
    """

    def __init__(
        self,
        manifest_path: str | Path = "data/annotations/work_manifest.json",
        processed_dir: str | Path = "data/processed",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.processed_dir = Path(processed_dir)
        self.works: dict[str, Work] = {}
        self.passages: list[Passage] = []
        self._term_frequencies: list[Counter[str]] = []
        self._document_frequencies: Counter[str] = Counter()
        self._document_lengths: list[int] = []
        self._passages_by_id: dict[str, Passage] = {}
        self._load()

    def _load(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        for item in manifest:
            work = Work(
                work_id=item["work_id"],
                title=item["title"],
                author=item["author"],
                language=item.get("language", ""),
            )
            self.works[work.work_id] = work
            chunks_path = self.processed_dir / f"{work.work_id}.chunks.json"
            if not chunks_path.exists():
                continue
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            for chunk in chunks:
                passage = Passage(
                    work_id=work.work_id,
                    work_title=work.title,
                    chapter=chunk.get("chapter", "?"),
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                )
                self.passages.append(passage)
                self._passages_by_id[passage.chunk_id] = passage
                frequencies = Counter(tokenize(passage.text))
                self._term_frequencies.append(frequencies)
                self._document_lengths.append(sum(frequencies.values()))
                self._document_frequencies.update(frequencies.keys())

        if not self.passages:
            raise ValueError("No chunk files found. Run the ingestion pipeline first.")

    @property
    def average_document_length(self) -> float:
        return sum(self._document_lengths) / max(len(self._document_lengths), 1)

    def get_passage(self, chunk_id: str) -> Passage | None:
        return self._passages_by_id.get(chunk_id)

    def search(
        self,
        query: str,
        work_ids: list[str] | None = None,
        top_k: int = 6,
    ) -> list[Passage]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        selected = set(work_ids or self.works.keys())
        total_documents = len(self.passages)
        average_length = self.average_document_length
        query_counts = Counter(query_tokens)
        scores: list[tuple[float, int]] = []
        k1, b = 1.5, 0.75

        for index, passage in enumerate(self.passages):
            if passage.work_id not in selected:
                continue
            frequencies = self._term_frequencies[index]
            document_length = self._document_lengths[index]
            score = 0.0
            matched_terms = 0
            for term, query_frequency in query_counts.items():
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                matched_terms += 1
                df = self._document_frequencies[term]
                inverse_document_frequency = math.log(1 + (total_documents - df + 0.5) / (df + 0.5))
                denominator = frequency + k1 * (1 - b + b * document_length / average_length)
                score += inverse_document_frequency * (frequency * (k1 + 1) / denominator) * min(query_frequency, 2)

            # Reward passages that cover several concepts in the question.
            score *= 1 + 0.15 * max(matched_terms - 1, 0)
            normalized_query = normalize_text(query)
            normalized_passage = normalize_text(passage.text)
            if len(normalized_query) > 8 and normalized_query in normalized_passage:
                score += 8.0
            if score > 0:
                scores.append((score, index))

        scores.sort(key=lambda item: item[0], reverse=True)
        return [
            Passage(**{**self.passages[index].__dict__, "score": round(score, 4)})
            for score, index in scores[:top_k]
        ]

    def stats(self) -> dict[str, int]:
        counts: defaultdict[str, int] = defaultdict(int)
        for passage in self.passages:
            counts[passage.work_id] += 1
        return dict(counts)
