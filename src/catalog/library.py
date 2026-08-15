from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ingestion.chunk_text import build_chunks_for_work
from ingestion.clean_text import clean_text
from ingestion.extract_epub import extract_epub_to_txt

from .models import CatalogBook, DownloadedBook


@dataclass(frozen=True)
class InstalledBook:
    work_id: str
    title: str
    already_installed: bool = False


class LibraryImporter:
    """Persist and index downloaded books in the user's local data folder."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.library_dir = self.data_dir / "library"
        self.catalog_path = self.library_dir / "installed_books.json"

    def install(self, book: CatalogBook, downloaded: DownloadedBook) -> InstalledBook:
        records = self._read_records()
        chunks_path = self.processed_dir / f"{book.work_id}.chunks.json"
        if any(record.get("work_id") == book.work_id for record in records) and chunks_path.exists():
            return InstalledBook(book.work_id, book.title, already_installed=True)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.library_dir.mkdir(parents=True, exist_ok=True)

        raw_path = self.raw_dir / f"{book.work_id}{downloaded.suffix}"
        extracted_path = self.processed_dir / f"{book.work_id}.extracted.txt"
        clean_path = self.processed_dir / f"{book.work_id}.clean.txt"
        raw_path.write_bytes(downloaded.content)

        if downloaded.suffix == ".epub":
            extract_epub_to_txt(raw_path, extracted_path)
            extracted = extracted_path.read_text(encoding="utf-8", errors="replace")
        else:
            extracted = self._decode_text(downloaded.content)
            extracted_path.write_text(extracted, encoding="utf-8")

        cleaned = clean_text(extracted)
        if len(cleaned.split()) < 100:
            raise ValueError("Le texte obtenu est trop court pour être indexé.")
        clean_path.write_text(cleaned, encoding="utf-8")
        build_chunks_for_work(book.work_id, clean_path, chunks_path)

        record = {
            "work_id": book.work_id,
            "title": book.title,
            "author": book.author_display,
            "language": book.languages[0] if book.languages else "",
            "source": "Project Gutenberg via Gutendex",
            "source_url": f"https://www.gutenberg.org/ebooks/{book.provider_id}",
            "provider": book.provider,
            "provider_id": book.provider_id,
            "raw_filename": raw_path.name,
            "download_url": downloaded.source_url,
        }
        records = [item for item in records if item.get("work_id") != book.work_id]
        records.append(record)
        self._write_records(records)
        return InstalledBook(book.work_id, book.title)

    def installed_ids(self) -> set[str]:
        return {str(record.get("work_id")) for record in self._read_records() if record.get("work_id")}

    def _read_records(self) -> list[dict]:
        if not self.catalog_path.exists():
            return []
        try:
            payload = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []

    def _write_records(self, records: list[dict]) -> None:
        self.library_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = self.catalog_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self.catalog_path)

    @staticmethod
    def _decode_text(content: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "windows-1252", "latin-1"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="replace")
