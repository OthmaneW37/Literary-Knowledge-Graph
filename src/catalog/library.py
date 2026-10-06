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
            "summary": "\n\n".join(book.summaries),
            "subjects": list(book.subjects),
            "metadata_sources": [{"source_name": "Project Gutenberg", "source_url": f"https://www.gutenberg.org/ebooks/{book.provider_id}",
                                  "summary": "\n\n".join(book.summaries), "subjects": list(book.subjects), "people": [],
                                  "retrieved_at": datetime.now(timezone.utc).isoformat()}],
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
        metadata: dict | None = None,
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
        if metadata:
            self._apply_metadata(record, metadata)
        installed = self._install_content(work_id, title, content, suffix, record)
        if metadata and installed.already_installed:
            self.attach_metadata(installed.work_id, metadata)
        return installed

    @staticmethod
    def _apply_metadata(record, metadata):
        sources = record.setdefault("metadata_sources", [])
        source = {key: metadata.get(key) for key in ("provider", "provider_id", "source_name", "source_url", "retrieved_at", "summary", "subjects", "people", "first_publish_year")}
        record["metadata_sources"] = [s for s in sources if s.get("source_url") != source["source_url"]] + [source]
        if metadata.get("summary"):
            record["summary"] = metadata["summary"]
        record["subjects"] = list(dict.fromkeys([*record.get("subjects", []), *metadata.get("subjects", [])]))[:50]
        if metadata.get("first_publish_year"):
            record["first_publish_year"] = metadata["first_publish_year"]
        if not str(record.get("cover_url", "")).startswith("/api/visuals/") and metadata.get("cover_url"):
            record["cover_url"] = metadata["cover_url"]
            record["cover_origin"] = metadata["source_name"]

    def attach_metadata(self, work_id, metadata):
        record = next((dict(r) for r in self.records() if r["work_id"] == work_id), None)
        if record is None:
            raise ValueError("Livre inconnu.")
        self._apply_metadata(record, metadata)
        records = [r for r in self._read_records() if r["work_id"] != work_id]
        self._write_records([*records, record])

    def _install_content(
        self,
        work_id: str,
        title: str,
        content: bytes,
        suffix: str,
        record: dict,
        force: bool = False,
    ) -> InstalledBook:
        records = self._read_records()
        chunks_path = self.processed_dir / f"{work_id}.chunks.json"
        visuals_path = self.processed_dir / f"{work_id}.visuals.json"
        if not force and any(item.get("work_id") == work_id for item in records) and chunks_path.exists():
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
            if document.summary and not record.get("summary"):
                record["summary"] = document.summary
                record.setdefault("metadata_sources", []).append({"source_name": "Métadonnées de l’EPUB",
                    "source_url": "", "summary": document.summary, "subjects": list(document.subjects), "people": []})
            record["subjects"] = list(dict.fromkeys([*record.get("subjects", []), *document.subjects]))[:50]
            visual_dir = self.library_dir / "visuals" / work_id
            staged_visual_dir = stage / "visuals"
            staged_visual_dir.mkdir(parents=True, exist_ok=True)
            visual_records = []
            for number, visual in enumerate(document.visuals, start=1):
                if not visual.content:
                    continue
                suffix_by_type = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}
                image_suffix = suffix_by_type.get(visual.media_type, ".bin")
                visual_id = hashlib.sha256(visual.content).hexdigest()[:20]
                filename = f"{visual_id}{image_suffix}"
                (staged_visual_dir / filename).write_bytes(visual.content)
                visual_records.append({
                    "visual_id": visual_id, "work_id": work_id, "chapter": visual.chapter,
                    "page": visual.page, "source_document": visual.source_document,
                    "type": visual.visual_type, "local_path": str((visual_dir / filename).resolve()),
                    "caption": visual.caption, "surrounding_text": visual.surrounding_text,
                    "ocr_text": visual.ocr_text,
                    "access": {"owner": "local_user", "visibility": "private", "stored_locally": True},
                    "media_type": visual.media_type,
                })
                if visual.visual_type == "cover":
                    record["cover_url"] = f"/api/visuals/{visual_id}"
                    record["cover_origin"] = "Couverture intégrée au fichier EPUB"
            if suffix == ".pdf":
                record["page_numbering"] = "physical_pdf_page"
            cleaned = clean_text(extracted)
            if len(cleaned.split()) < 100:
                raise ValueError(
                    "Le texte obtenu est trop court. Le PDF est peut-être scanné sans couche de texte."
                )
            staged_clean = stage / clean_path.name
            staged_clean.write_text(cleaned, encoding="utf-8")
            (stage / visuals_path.name).write_text(json.dumps(visual_records, ensure_ascii=False, indent=2), encoding="utf-8")
            build_chunks_for_work(
                work_id, staged_clean, stage / chunks_path.name,
                chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "240")),
                overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "40")),
            )
            (stage / raw_path.name).write_bytes(content)
            (stage / extracted_path.name).write_text(extracted, encoding="utf-8")
            for target in (raw_path, extracted_path, clean_path, chunks_path, visuals_path):
                (stage / target.name).replace(target)
            if visual_dir.exists():
                import shutil
                shutil.rmtree(visual_dir)
            if staged_visual_dir.exists() and any(staged_visual_dir.iterdir()):
                visual_dir.parent.mkdir(parents=True, exist_ok=True)
                staged_visual_dir.replace(visual_dir)
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
        return self._install_content(
            work_id, record.get("title", ""), source.read_bytes(), source.suffix.casefold(),
            dict(record), force=True,
        )

    def _read_records(self) -> list[dict]:
        if not self.catalog_path.exists():
            return []
        payload = read_json(self.catalog_path, [])
        return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []

    def _write_records(self, records: list[dict]) -> None:
        write_json(self.catalog_path, records)
