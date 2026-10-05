from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

from ingestion.chunk_text import build_chunks_for_work
from ingestion.clean_text import clean_text
from ingestion.loaders import DocumentLoader, detect_language
from storage.local import read_json, write_json

from .models import CatalogBook, DownloadedBook


@dataclass(frozen=True)
class InstalledBook:
    work_id: str
    title: str
    already_installed: bool = False


class LibraryImporter:
    """Persist and index downloaded books in the user's local data folder."""

    MAX_UPLOAD_BYTES = 50 * 1024 * 1024
    SUPPORTED_UPLOAD_SUFFIXES = DocumentLoader.SUPPORTED_SUFFIXES

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.library_dir = self.data_dir / "library"
        self.catalog_path = self.library_dir / "installed_books.json"

    def install(self, book: CatalogBook, downloaded: DownloadedBook) -> InstalledBook:
        record = {
            "work_id": book.work_id,
            "title": book.title,
            "author": book.author_display,
            "language": book.languages[0] if book.languages else "",
            "source": "Project Gutenberg via Gutendex",
            "source_url": f"https://www.gutenberg.org/ebooks/{book.provider_id}",
            "provider": book.provider,
            "provider_id": book.provider_id,
            "raw_filename": f"{book.work_id}{downloaded.suffix}",
            "download_url": downloaded.source_url,
            "cover_url": book.formats.get("image/jpeg", ""),
        }
        return self._install_content(
            work_id=book.work_id,
            title=book.title,
            content=downloaded.content,
            suffix=downloaded.suffix,
            record=record,
        )

    def install_upload(
        self,
        content: bytes,
        filename: str,
        title: str = "",
        author: str = "Auteur inconnu",
        language: str = "",
    ) -> InstalledBook:
        """Index a legally obtained EPUB, TXT, Markdown or text-based PDF."""
        suffix = Path(filename).suffix.casefold()
        if suffix not in self.SUPPORTED_UPLOAD_SUFFIXES:
            raise ValueError(
                "Format non pris en charge. Utilisez EPUB, TXT, Markdown, PDF ou DOCX."
            )
        if not content:
            raise ValueError("Le fichier envoyé est vide.")
        if len(content) > self.MAX_UPLOAD_BYTES:
            raise ValueError("Le fichier dépasse la limite de 50 Mo.")
        title = title.strip()

        digest = hashlib.sha256(content).hexdigest()[:16]
        work_id = f"upload_{digest}"
        record = {
            "work_id": work_id,
            "title": title,
            "author": author.strip() or "Auteur inconnu",
            "language": language.strip().casefold(),
            "source": "Fichier fourni par l'utilisateur",
            "source_url": "",
            "provider": "upload",
            "provider_id": digest,
            "raw_filename": f"{work_id}{suffix}",
            "original_filename": Path(filename).name,
        }
        return self._install_content(work_id, title, content, suffix, record)

    def _install_content(
        self,
        work_id: str,
        title: str,
        content: bytes,
        suffix: str,
        record: dict,
    ) -> InstalledBook:
        records = self._read_records()
        chunks_path = self.processed_dir / f"{work_id}.chunks.json"
        if any(item.get("work_id") == work_id for item in records) and chunks_path.exists():
            self.set_archived(work_id, False)
            stored = next(item for item in records if item.get("work_id") == work_id)
            return InstalledBook(work_id, stored["title"], already_installed=True)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        raw_path = self.raw_dir / f"{work_id}{suffix}"
        extracted_path = self.processed_dir / f"{work_id}.extracted.txt"
        clean_path = self.processed_dir / f"{work_id}.clean.txt"
        # Parse and validate before replacing any previously installed files.
        with tempfile.TemporaryDirectory(dir=self.library_dir) as temporary:
            stage = Path(temporary)
            document = DocumentLoader().load(raw_path, content=content)
            extracted = document.text
            record["title"] = record.get("title") or document.title or title or Path(record.get("original_filename", work_id)).stem
            record["author"] = (
                document.author
                if record.get("author") in {"", "Auteur inconnu"} and document.author
                else record.get("author", "Auteur inconnu")
            )
            record["language"] = (
                record.get("language")
                or document.language.split("-")[0].casefold()
                or detect_language(extracted)
            )
            record["page_count"] = document.page_count
            if suffix == ".pdf":
                record["page_numbering"] = "physical_pdf_page"
            cleaned = clean_text(extracted)
            if len(cleaned.split()) < 100:
                raise ValueError(
                    "Le texte obtenu est trop court. Le PDF est peut-être scanné sans couche de texte."
                )
            staged_clean = stage / clean_path.name
            staged_clean.write_text(cleaned, encoding="utf-8")
            build_chunks_for_work(
                work_id, staged_clean, stage / chunks_path.name,
                chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "240")),
                overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "40")),
            )
            (stage / raw_path.name).write_bytes(content)
            (stage / extracted_path.name).write_text(extracted, encoding="utf-8")
            for target in (raw_path, extracted_path, clean_path, chunks_path):
                (stage / target.name).replace(target)
        record["imported_at"] = record.get("imported_at") or datetime.now(timezone.utc).isoformat()
        record["content_hash"] = hashlib.sha256(content).hexdigest()
        record["size_bytes"] = len(content)
        record["index_status"] = "text_ready"

        records = [item for item in records if item.get("work_id") != work_id]
        records.append(record)
        self._write_records(records)
        self.set_archived(work_id, False)
        return InstalledBook(work_id, str(record.get("title") or title))

    def installed_ids(self) -> set[str]:
        return {str(record.get("work_id")) for record in self._read_records() if record.get("work_id")} - self.archived_ids()

    def records(self) -> list[dict]:
        bundled = read_json(self.data_dir / "annotations/work_manifest.json", [])
        return list({record["work_id"]: record for record in [*bundled, *self._read_records()]}.values())

    def archived_ids(self) -> set[str]:
        return set(read_json(self.library_dir / "archived_books.json", []))

    def set_archived(self, work_id: str, archived: bool = True) -> None:
        if not any(record["work_id"] == work_id for record in self.records()):
            raise ValueError("Livre inconnu.")
        ids = self.archived_ids()
        ids.add(work_id) if archived else ids.discard(work_id)
        write_json(self.library_dir / "archived_books.json", sorted(ids))

    def reindex(self, work_id: str) -> InstalledBook:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", work_id):
            raise ValueError("Identifiant de livre invalide.")
        record = next((item.copy() for item in self.records() if item["work_id"] == work_id), None)
        if record is None:
            raise ValueError("Livre inconnu.")
        raw_name = record.get("raw_filename", f"{work_id}.epub")
        raw_path = self.raw_dir / Path(raw_name).name
        if not raw_path.is_file():
            raise ValueError("Fichier original absent ; importez à nouveau votre exemplaire.")
        # Bypass the idempotent import check without deleting the current index.
        return self._rebuild_record(record, raw_path)

    def _rebuild_record(self, record: dict, source: Path) -> InstalledBook:
        work_id = record["work_id"]
        document = DocumentLoader().load(source)
        cleaned = clean_text(document.text)
        if len(cleaned.split()) < 100:
            raise ValueError("Texte trop court pour réindexer ce livre.")
        self.library_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.library_dir) as temporary:
            path = Path(temporary) / f"{work_id}.clean.txt"
            path.write_text(cleaned, encoding="utf-8")
            output = path.with_name(f"{work_id}.chunks.json")
            build_chunks_for_work(work_id, path, output, int(os.getenv("RAG_CHUNK_SIZE", "240")), int(os.getenv("RAG_CHUNK_OVERLAP", "40")))
            output.replace(self.processed_dir / output.name)
            path.replace(self.processed_dir / path.name)
        return InstalledBook(work_id, record["title"])

    def _read_records(self) -> list[dict]:
        if not self.catalog_path.exists():
            return []
        payload = read_json(self.catalog_path, [])
        return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []

    def _write_records(self, records: list[dict]) -> None:
        write_json(self.catalog_path, records)
