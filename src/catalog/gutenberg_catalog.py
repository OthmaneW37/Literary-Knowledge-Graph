"""Local index of the official Project Gutenberg catalog.

Gutendex, the community JSON mirror, is often unreachable. gutenberg.org
publishes its whole catalog (about 80 000 public-domain books) as one CSV
file. It is downloaded once, cached under data/library/catalog and refreshed
weekly, then searched in memory: any public-domain book is found in
milliseconds and its EPUB or text is fetched directly from gutenberg.org.
"""
from __future__ import annotations

import csv
import difflib
import gzip
import io
import logging
import math
import os
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from threading import Lock

import httpx

from .gutendex import CatalogError, GutendexClient
from .models import CatalogBook, CatalogPage

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
MAX_AGE_SECONDS = 7 * 24 * 3600
# Words that never identify a book, in the languages readers type queries in.
STOP_WORDS = {
    "the", "a", "an", "of", "and", "in", "on", "to", "by", "de", "des", "du", "la", "le", "les", "l", "d", "un", "une",
    "et", "en", "au", "aux", "el", "los", "las", "del", "y", "der", "die", "das", "und", "von", "il", "lo", "gli", "di",
    "book", "livre", "roman", "novel", "ebook",
}


def words(value: str) -> list[str]:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return [word for word in re.findall(r"[a-z0-9]+", value) if word not in STOP_WORDS and (len(word) > 1 or word.isdigit())]


class GutenbergCatalog:
    """Search, look up and download any public-domain Gutenberg book."""

    def __init__(self, cache_dir: str | Path, client: httpx.Client | None = None,
                 downloader: GutendexClient | None = None, url: str | None = None) -> None:
        self.cache_path = Path(cache_dir) / "pg_catalog.csv.gz"
        self.url = url or os.getenv("GUTENBERG_CATALOG_URL", CATALOG_URL)
        self.client = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0), follow_redirects=True,
                                             headers={"User-Agent": "NarrativeLens/2.0 (local reading assistant)"})
        # Gutendex's download routine already restricts URLs to gutenberg.org.
        self.downloader = downloader or GutendexClient(client=self.client)
        self._lock = Lock()
        self._books: dict[int, dict] = {}
        self._postings: dict[str, set[int]] = defaultdict(set)
        self._vocabulary_by_prefix: dict[str, set[str]] = defaultdict(set)
        self._loaded_at = 0.0

    # Loading ---------------------------------------------------------------
    def ensure_loaded(self) -> None:
        with self._lock:
            if self._books and time.time() - self._loaded_at < MAX_AGE_SECONDS:
                return
            stale = not self.cache_path.exists() or time.time() - self.cache_path.stat().st_mtime > MAX_AGE_SECONDS
            if stale:
                try:
                    self._download()
                except (httpx.HTTPError, OSError) as exc:
                    if not self.cache_path.exists():
                        raise CatalogError("Le catalogue Project Gutenberg est momentanément inaccessible.") from exc
                    logging.warning("Using the cached Gutenberg catalog; refresh failed: %s", exc)
            if not self._books or stale:
                self._index(self.cache_path.read_bytes())

    def _download(self) -> None:
        response = self.client.get(self.url)
        response.raise_for_status()
        content = response.content
        if not content.startswith(b"\x1f\x8b"):
            raise OSError("The Gutenberg catalog is not a gzip file")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(self.cache_path)

    def _index(self, compressed: bytes) -> None:
        books: dict[int, dict] = {}
        postings: dict[str, set[int]] = defaultdict(set)
        with gzip.open(io.BytesIO(compressed), "rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("Type") != "Text" or not str(row.get("Text#", "")).isdigit():
                    continue
                book_id = int(row["Text#"])
                authors = [a.strip() for a in row.get("Authors", "").split(";") if a.strip()]
                main = [a for a in authors if not re.search(r"\[(?:Translator|Editor|Illustrator|Contributor)", a)]
                record = {
                    "id": book_id, "title": " ".join(row.get("Title", "").split()) or "Sans titre",
                    "authors": [self._display_author(a) for a in (main or authors)],
                    "languages": [code.strip() for code in row.get("Language", "").split(";") if code.strip()],
                    "subjects": [s.strip() for s in row.get("Subjects", "").split(";") if s.strip()][:8],
                    "shelves": row.get("Bookshelves", ""),
                }
                title_words = set(words(record["title"]))
                author_words = set(words(" ".join(main or authors)))
                record["_title"] = title_words
                books[book_id] = record
                for word in title_words | author_words:
                    postings[word].add(book_id)
        vocabulary = defaultdict(set)
        for word in postings:
            if len(word) >= 4:
                vocabulary[word[:3]].add(word)
        self._books, self._postings, self._vocabulary_by_prefix = books, postings, vocabulary
        self._loaded_at = time.time()

    @staticmethod
    def _display_author(author: str) -> str:
        """"Dostoyevsky, Fyodor, 1821-1881" -> "Fyodor Dostoyevsky"."""
        author = re.sub(r"\s*\[[^\]]*\]$", "", author)
        # Drop the trailing life dates, whatever their form ("1821-1881", "44 BCE-18? CE").
        author = re.sub(r",[^,]*\d[^,]*$", "", author).strip(" ,")
        parts = [part.strip() for part in author.split(",")]
        return f"{parts[1]} {parts[0]}" if len(parts) == 2 and parts[1] else author

    # Search ----------------------------------------------------------------
    def _expand(self, word: str) -> dict[str, float]:
        """Exact word, plus close spellings for transliterated names (Dostoïevski/Dostoyevsky)."""
        matches = {word: 1.0} if word in self._postings else {}
        if len(word) >= 5:
            for candidate in self._vocabulary_by_prefix.get(word[:3], ()):
                # A shorter word is another name, not a spelling ("Bova" is not "Bovary").
                if candidate != word and len(candidate) >= len(word) - 1:
                    ratio = difflib.SequenceMatcher(None, word, candidate).ratio()
                    if ratio >= 0.78:
                        matches[candidate] = max(matches.get(candidate, 0.0), ratio * 0.9)
        return matches

    def search(self, query: str, language: str | None = None, page: int = 1, limit: int = 12) -> CatalogPage:
        query = query.strip()
        if len(query) < 2:
            raise ValueError("La recherche doit contenir au moins 2 caractères.")
        self.ensure_loaded()
        terms = words(query)
        if not terms:
            return CatalogPage(count=0, books=(), page=page)
        total = max(len(self._books), 1)
        scores: dict[int, float] = defaultdict(float)
        hits: dict[int, int] = defaultdict(int)
        matched_by_term: list[set[int]] = []
        for term in terms:
            best: dict[int, float] = {}
            for variant, weight in self._expand(term).items():
                ids = self._postings[variant]
                idf = math.log(1 + total / (1 + len(ids)))
                for book_id in ids:
                    boost = 1.4 if variant in self._books[book_id]["_title"] else 1.0
                    best[book_id] = max(best.get(book_id, 0.0), weight * idf * boost)
            for book_id, score in best.items():
                scores[book_id] += score
                hits[book_id] += 1
            if best:
                matched_by_term.append(set(best))
        # Most meaningful words must match, except translated title words
        # ("frères Karamazov" still finds "The Brothers Karamazov").
        needed = 1 if len(terms) <= 2 else math.ceil(len(terms) / 2)
        candidates = [book_id for book_id in scores if hits[book_id] >= needed]
        # A rare word ("Karamazov") identifies the book; common words that
        # only matched because of the translation ("frères") must not dilute it.
        rarest = min(matched_by_term, key=len, default=set())
        best = max(candidates, key=lambda book_id: scores[book_id], default=None)
        if best in rarest and len(rarest) <= 200 and len(matched_by_term) > 1:
            candidates = [book_id for book_id in candidates if book_id in rarest]
        if language:
            candidates = [book_id for book_id in candidates if language in self._books[book_id]["languages"]]

        def rank(book_id):
            shelves = self._books[book_id]["shelves"]
            popularity = 2.0 * ("Best Books Ever" in shelves) + 1.0 * ("Classics of Literature" in shelves) + 0.5 * bool(shelves)
            # The interface is French and most curated editions are English.
            popularity += 0.6 * bool({"fr", "en"} & set(self._books[book_id]["languages"]))
            return -(scores[book_id] + popularity), book_id

        candidates.sort(key=rank)
        start = (max(page, 1) - 1) * limit
        books = tuple(self._catalog_book(self._books[book_id]) for book_id in candidates[start:start + limit])
        return CatalogPage(count=len(candidates), books=books, page=max(page, 1),
                           has_next=start + limit < len(candidates), has_previous=page > 1)

    def get_book(self, provider_id: int) -> CatalogBook:
        if provider_id <= 0:
            raise ValueError("L'identifiant Gutenberg doit être positif.")
        try:
            self.ensure_loaded()
        except CatalogError:
            return self.downloader.get_book(provider_id)
        record = self._books.get(provider_id)
        return self._catalog_book(record) if record else self.downloader.get_book(provider_id)

    def download(self, book: CatalogBook):
        return self.downloader.download(book)

    @staticmethod
    def _catalog_book(record: dict) -> CatalogBook:
        book_id = record["id"]
        base = f"https://www.gutenberg.org/ebooks/{book_id}"
        return CatalogBook(
            provider_id=book_id, title=record["title"], authors=tuple(record["authors"]),
            languages=tuple(record["languages"]), subjects=tuple(record["subjects"]),
            formats={"application/epub+zip": f"{base}.epub3.images",
                     "text/plain; charset=utf-8": f"{base}.txt.utf-8",
                     "image/jpeg": f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.cover.medium.jpg"},
            copyright=False,
        )
