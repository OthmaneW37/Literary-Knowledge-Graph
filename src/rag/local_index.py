from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .models import Passage, Work
from storage.local import read_json


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
    # Keep significant initials such as "K." while still ignoring single-letter noise.
    value = re.sub(r"\b([A-Z])\.(?=\W|$)", lambda match: f" initial{match[1].lower()} ", value)
    tokens = TOKEN_PATTERN.findall(normalize_text(value))
    normalized_tokens: list[str] = []
    for token in tokens:
        token = token.replace("’", "'")
        if token.endswith("'s"):
            token = token[:-2]
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if len(token) > 1 and token not in STOP_WORDS:
            normalized_tokens.append(token)
    return normalized_tokens


class LocalLiteraryIndex:
    """In-memory BM25 index over locally processed novels.

    An inverted term index avoids scanning the entire library for each query.
    The public search API can later be complemented by semantic retrieval.
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
        self._passage_positions: dict[str, int] = {}
        self._postings: defaultdict[str, list[int]] = defaultdict(list)
        self._load()

    def _load(self) -> None:
        manifest = self._load_manifest_records()
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
                    author=work.author,
                    language=work.language,
                    section=str(chunk.get("section", "")),
                    page=chunk.get("page"),
                    pages=tuple(chunk.get("pages", [])),
                    chunk_index=int(chunk.get("chunk_index", 0) or 0),
                    start_char=int(chunk.get("start_char", 0) or 0),
                    end_char=int(chunk.get("end_char", 0) or 0),
                )
                self.passages.append(passage)
                self._passages_by_id[passage.chunk_id] = passage
                self._passage_positions[passage.chunk_id] = len(self.passages) - 1
                frequencies = Counter(tokenize(passage.text))
                self._term_frequencies.append(frequencies)
                self._document_lengths.append(sum(frequencies.values()))
                self._document_frequencies.update(frequencies.keys())
                passage_index = len(self.passages) - 1
                for term in frequencies:
                    self._postings[term].append(passage_index)


    def _load_manifest_records(self) -> list[dict]:
        """Merge bundled works with books installed in the local library."""
        paths = [self.manifest_path]
        local_catalog = self.manifest_path.parent.parent / "library" / "installed_books.json"
        if local_catalog.exists():
            paths.append(local_catalog)

        records_by_id: dict[str, dict] = {}
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, list):
                continue
            for item in payload:
                if isinstance(item, dict) and item.get("work_id"):
                    records_by_id[str(item["work_id"])] = item
        archived = set(read_json(self.manifest_path.parent.parent / "library/archived_books.json", []))
        return [record for work_id, record in records_by_id.items() if work_id not in archived]

    @property
    def average_document_length(self) -> float:
        return sum(self._document_lengths) / max(len(self._document_lengths), 1)

    def get_passage(self, chunk_id: str) -> Passage | None:
        return self._passages_by_id.get(chunk_id)

    def get_neighbors(self, chunk_id: str, radius: int = 1) -> list[Passage]:
        """Return adjacent passages from the same work in narrative order."""
        position = self._passage_positions.get(chunk_id)
        if position is None or radius < 1:
            return []
        passage = self.passages[position]
        start = max(0, position - radius)
        end = min(len(self.passages), position + radius + 1)
        return [
            candidate
            for candidate in self.passages[start:end]
            if candidate.work_id == passage.work_id and candidate.chunk_id != chunk_id
        ]

    def search(
        self,
        query: str,
        work_ids: list[str] | None = None,
        top_k: int = 6,
        max_chapter: int | None = None,
    ) -> list[Passage]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        selected = set(self.works if work_ids is None else work_ids)
        total_documents = len(self.passages)
        average_length = self.average_document_length
        query_counts = Counter(query_tokens)
        beginning_query = bool({"beginn", "start", "open", "debut"} & set(query_tokens))
        scores: list[tuple[float, int]] = []
        k1, b = 1.5, 0.75

        candidate_indices = {
            passage_index
            for term in query_counts
            for passage_index in self._postings.get(term, [])
        }
        for index in candidate_indices:
            passage = self.passages[index]
            if passage.work_id not in selected:
                continue
            if max_chapter is not None:
                try:
                    if int(passage.chapter) > max_chapter:
                        continue
                except (TypeError, ValueError):
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
            try:
                chapter_number = int(passage.chapter)
            except (TypeError, ValueError):
                chapter_number = 0
            if beginning_query and chapter_number == 1:
                score += 3.0
            if score > 0:
                scores.append((score, index))

        scores.sort(key=lambda item: (-item[0], item[1]))
        return [
            Passage(**{**self.passages[index].__dict__, "score": round(score, 4)})
            for score, index in scores[:top_k]
        ]

    def stats(self) -> dict[str, int]:
        counts: defaultdict[str, int] = defaultdict(int)
        for passage in self.passages:
            counts[passage.work_id] += 1
        return dict(counts)
