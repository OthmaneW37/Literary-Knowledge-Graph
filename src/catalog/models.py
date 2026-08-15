from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CatalogBook:
    provider_id: int
    title: str
    authors: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    subjects: tuple[str, ...] = ()
    summaries: tuple[str, ...] = ()
    formats: dict[str, str] = field(default_factory=dict)
    download_count: int = 0
    copyright: bool | None = None
    provider: str = "gutendex"

    @property
    def work_id(self) -> str:
        return f"gutenberg_{self.provider_id}"

    @property
    def author_display(self) -> str:
        return ", ".join(self.authors) if self.authors else "Auteur inconnu"

    @property
    def language_display(self) -> str:
        return ", ".join(language.upper() for language in self.languages) or "?"


@dataclass(frozen=True)
class CatalogPage:
    count: int
    books: tuple[CatalogBook, ...]
    page: int = 1
    has_next: bool = False
    has_previous: bool = False


@dataclass(frozen=True)
class DownloadedBook:
    content: bytes
    mime_type: str
    suffix: str
    source_url: str
