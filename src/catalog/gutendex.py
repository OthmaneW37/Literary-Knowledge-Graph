from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import httpx

from .models import CatalogBook, CatalogPage, DownloadedBook


DEFAULT_BASE_URL = "https://gutendex.com"
MAX_BOOK_BYTES = 25 * 1024 * 1024
SUPPORTED_FORMATS = (
    ("application/epub+zip", ".epub"),
    ("text/plain; charset=utf-8", ".txt"),
    ("text/plain; charset=us-ascii", ".txt"),
    ("text/plain", ".txt"),
)


class CatalogError(RuntimeError):
    """A remote catalog request or book download failed."""


class GutendexClient:
    """Small client for the public Gutendex Project Gutenberg catalog."""

    def __init__(
        self,
        base_url: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("GUTENDEX_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(25.0, connect=8.0),
            follow_redirects=True,
            headers={"User-Agent": "Literary-Knowledge-Graph/0.1"},
        )

    def close(self) -> None:
        self.client.close()

    def search(
        self,
        query: str,
        language: str | None = None,
        page: int = 1,
        limit: int = 12,
    ) -> CatalogPage:
        query = query.strip()
        if len(query) < 2:
            raise ValueError("La recherche doit contenir au moins 2 caractères.")
        params: dict[str, Any] = {
            "search": query,
            "page": max(page, 1),
            "copyright": "false,null",
        }
        if language:
            params["languages"] = language
        payload = self._get_json("/books", params=params)
        books = tuple(self._parse_book(item) for item in payload.get("results", [])[:limit])
        return CatalogPage(
            count=int(payload.get("count", len(books))),
            books=books,
            page=max(page, 1),
            has_next=bool(payload.get("next")),
            has_previous=bool(payload.get("previous")),
        )

    def get_book(self, provider_id: int) -> CatalogBook:
        if provider_id <= 0:
            raise ValueError("L'identifiant Gutenberg doit être positif.")
        return self._parse_book(self._get_json(f"/books/{provider_id}"))

    def download(self, book: CatalogBook) -> DownloadedBook:
        mime_type, suffix, url = self._select_format(book.formats)
        self._validate_download_url(url)
        try:
            response = self.client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise CatalogError(f"Téléchargement impossible pour « {book.title} ».") from exc
        content_length = int(response.headers.get("content-length", 0) or 0)
        if content_length > MAX_BOOK_BYTES or len(response.content) > MAX_BOOK_BYTES:
            raise CatalogError("Le fichier dépasse la limite de 25 Mo.")
        if not response.content:
            raise CatalogError("Le fichier téléchargé est vide.")
        self._validate_download_url(str(response.url))
        return DownloadedBook(response.content, mime_type, suffix, str(response.url))

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = self.client.get(f"{self.base_url}{path}", params=params)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise CatalogError("Le catalogue Gutendex est temporairement indisponible.") from exc
        if not isinstance(payload, dict):
            raise CatalogError("Réponse inattendue du catalogue Gutendex.")
        return payload

    @staticmethod
    def _parse_book(payload: dict[str, Any]) -> CatalogBook:
        authors = tuple(
            str(author.get("name", "")).strip()
            for author in payload.get("authors", [])
            if isinstance(author, dict) and str(author.get("name", "")).strip()
        )
        return CatalogBook(
            provider_id=int(payload["id"]),
            title=str(payload.get("title", "Sans titre")).strip() or "Sans titre",
            authors=authors,
            languages=tuple(str(value) for value in payload.get("languages", []) if value),
            subjects=tuple(str(value) for value in payload.get("subjects", []) if value),
            summaries=tuple(str(value) for value in payload.get("summaries", []) if value),
            formats={str(key): str(value) for key, value in payload.get("formats", {}).items() if value},
            download_count=int(payload.get("download_count", 0) or 0),
            copyright=payload.get("copyright"),
        )

    @staticmethod
    def _select_format(formats: dict[str, str]) -> tuple[str, str, str]:
        for wanted_mime, suffix in SUPPORTED_FORMATS:
            for actual_mime, url in formats.items():
                if actual_mime.casefold() == wanted_mime and url:
                    return actual_mime, suffix, url
        raise CatalogError("Ce livre ne propose ni EPUB ni texte brut compatible.")

    @staticmethod
    def _validate_download_url(url: str) -> None:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").casefold()
        if parsed.scheme != "https" or not (
            hostname == "gutenberg.org" or hostname.endswith(".gutenberg.org")
        ):
            raise CatalogError("Gutendex a fourni une adresse de téléchargement non autorisée.")
