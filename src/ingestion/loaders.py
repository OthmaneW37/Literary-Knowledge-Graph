from __future__ import annotations

import re
import os
import posixpath
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from bs4 import BeautifulSoup


class DocumentLoadError(ValueError):
    """A document cannot be converted into useful, searchable text."""


@dataclass(frozen=True)
class ExtractedVisual:
    """An image extracted from a book with its nearest reading context."""

    content: bytes
    media_type: str
    chapter: int | None = None
    page: int | None = None
    source_document: str = ""
    visual_type: str = "illustration"
    caption: str = ""
    surrounding_text: str = ""
    ocr_text: str = ""


@dataclass(frozen=True)
class LoadedDocument:
    text: str
    title: str = ""
    author: str = ""
    language: str = ""
    page_count: int = 0
    likely_scanned: bool = False
    visuals: tuple[ExtractedVisual, ...] = ()
    summary: str = ""
    subjects: tuple[str, ...] = ()


class DocumentLoader:
    """Format-aware local document extraction with one normalized result type."""

    SUPPORTED_SUFFIXES = {".epub", ".txt", ".md", ".pdf", ".docx"}
    MAX_EXTRACTED_CHARS = 8_000_000
    MIN_PDF_TEXT_CHARS = 100

    def load(self, path: str | Path, content: bytes | None = None) -> LoadedDocument:
        source = Path(path)
        suffix = source.suffix.casefold()
        if suffix not in self.SUPPORTED_SUFFIXES:
            raise DocumentLoadError(
                f"Format non pris en charge ({suffix or 'inconnu'}). Utilisez EPUB, TXT, Markdown, PDF ou DOCX."
            )
        if content is None:
            try:
                content = source.read_bytes()
            except OSError as exc:
                raise DocumentLoadError("Impossible de lire le fichier fourni.") from exc
        if not content:
            raise DocumentLoadError("Le fichier est vide.")

        if suffix == ".pdf":
            document = self._load_pdf(content)
        elif suffix == ".epub":
            document = self._load_epub(content)
        elif suffix == ".docx":
            document = self._load_docx(content)
        else:
            document = LoadedDocument(self._decode_text(content))

        if len(document.text) > self.MAX_EXTRACTED_CHARS:
            raise DocumentLoadError("Le texte extrait dépasse la limite de sécurité de 8 millions de caractères.")
        if document.likely_scanned or len(document.text.strip()) < self.MIN_PDF_TEXT_CHARS:
            if document.likely_scanned:
                raise DocumentLoadError(
                    "Ce PDF contient très peu de texte exploitable et semble scanné. "
                    "Une étape OCR locale sera nécessaire avant son import."
                )
            raise DocumentLoadError("Le document ne contient pas assez de texte pour être indexé.")
        return document

    @staticmethod
    def _decode_text(content: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "windows-1252", "latin-1"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="replace")

    @staticmethod
    def _load_pdf(content: bytes) -> LoadedDocument:
        from pypdf import PdfReader

        try:
            reader = PdfReader(BytesIO(content))
            sections = []
            ocr_sections = []
            visuals = []
            page_text_lengths = []
            for number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                page_text_lengths.append(len(text.strip()))
                if text.strip():
                    sections.append(f"[[PAGE:{number}]]\n{text.strip()}")
                # pypdf exposes embedded images without rendering every page.
                # Broken or unsupported image objects do not invalidate text ingestion.
                try:
                    for image in page.images:
                        image_bytes = image.data
                        if image_bytes:
                            suffix = Path(image.name).suffix.casefold()
                            media_type = {
                                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                                ".png": "image/png", ".webp": "image/webp",
                            }.get(suffix, "application/octet-stream")
                            visuals.append(ExtractedVisual(
                                image_bytes, media_type, page=number,
                                source_document=f"page-{number}",
                                visual_type="page" if len(text.strip()) < 80 else "illustration",
                                surrounding_text=text.strip()[:1200],
                            ))
                            if os.getenv("OCR_ENABLED", "false").casefold() in {"1", "true", "yes"} and len(text.strip()) < 35:
                                try:
                                    from PIL import Image
                                    import pytesseract
                                    recognized = pytesseract.image_to_string(Image.open(BytesIO(image_bytes))).strip()
                                    if recognized:
                                        ocr_sections.append(f"[[PAGE:{number}]]\n{recognized}")
                                        visuals[-1] = ExtractedVisual(
                                            image_bytes, media_type, page=number,
                                            source_document=f"page-{number}", visual_type="page",
                                            surrounding_text=recognized[:1200], ocr_text=recognized[:8000],
                                        )
                                except (ImportError, OSError, RuntimeError):
                                    # OCR is deliberately optional; searchable
                                    # text PDFs must not depend on it.
                                    pass
                except Exception:
                    pass
        except Exception as exc:
            raise DocumentLoadError("Le PDF est invalide ou endommagé.") from exc
        text = "\n\n".join(sections or ocr_sections)
        likely_scanned = not text.strip() or (
            len(text.strip()) < DocumentLoader.MIN_PDF_TEXT_CHARS
            or (len(page_text_lengths) >= 2 and sum(page_text_lengths) / len(page_text_lengths) < 35)
        )
        return LoadedDocument(text, page_count=len(page_text_lengths), likely_scanned=likely_scanned, visuals=tuple(visuals))

    @staticmethod
    def _load_epub(content: bytes) -> LoadedDocument:
        from ebooklib import ITEM_DOCUMENT, ITEM_COVER, epub
        from zipfile import ZipFile

        try:
            with ZipFile(BytesIO(content)) as zipped:
                if len(zipped.infolist()) > 5000 or sum(i.file_size for i in zipped.infolist()) > 150 * 1024 * 1024:
                    raise DocumentLoadError("L’EPUB dépasse la taille décompressée autorisée.")
            with BytesIO(content) as archive:
                book = epub.read_epub(archive)
            parts = []
            visuals = []
            cover_ids = {attrs.get("content") for _, attrs in book.get_metadata("OPF", "cover")}
            for image in book.get_items():
                if image.get_type() == ITEM_COVER or "cover-image" in getattr(image, "properties", []) or image.id in cover_ids:
                    if getattr(image, "media_type", "") in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
                        visuals.append(ExtractedVisual(image.get_content(), image.media_type,
                            source_document=image.file_name, visual_type="cover", caption="Couverture de l’EPUB"))
            # EPUB spine, not archive item order, defines reading order.
            items = [book.get_item_with_id(item_id) for item_id, linear in book.spine if linear != "no"]
            if not items:
                items = list(book.get_items_of_type(ITEM_DOCUMENT))
            for item in items:
                filename = Path(item.file_name).stem.casefold() if item is not None else ""
                item_id = str(getattr(item, "id", "")).casefold() if item is not None else ""
                # Project Gutenberg wraps novels with a repeated legal/TOC
                # header and footer in the reading spine. These are not part
                # of the story and can otherwise shift every chapter number.
                boilerplate = re.search(
                    r"(?:^|[-_\s])(?:pg-header|pg-footer|coverpage-wrapper|titlepage|toc|nav)(?:[-_\s]|$)",
                    f"{item_id} {filename}",
                )
                if (
                    item is not None
                    and item.get_type() == ITEM_DOCUMENT
                    and "nav" not in item.properties
                    and not boilerplate
                ):
                    soup = BeautifulSoup(item.get_content(), "html.parser")
                    for element in soup(["script", "style", "nav"]):
                        element.decompose()
                    for element in soup.find_all(["p", "h1", "h2", "h3", "div", "blockquote", "li"]):
                        element.insert_after("\n\n")
                    text = soup.get_text().strip()
                    if text:
                        parts.append(text)
                    for img in soup.find_all("img"):
                        image_href = posixpath.normpath(posixpath.join(posixpath.dirname(item.file_name), img.get("src", "")))
                        image_item = book.get_item_with_href(image_href)
                        if image_item is None:
                            image_item = next((candidate for candidate in book.get_items() if candidate.file_name == image_href), None)
                        if image_item is None:
                            continue
                        image_bytes = image_item.get_content()
                        media_type = getattr(image_item, "media_type", "") or "application/octet-stream"
                        caption = img.get("alt", "").strip()
                        visuals.append(ExtractedVisual(
                            image_bytes, media_type, chapter=len(parts) or None,
                            source_document=str(item.file_name),
                            visual_type="map" if any(word in caption.casefold() for word in ("map", "carte")) else "illustration",
                            caption=caption, surrounding_text=text[:1200],
                        ))
            metadata = lambda key: next(
                (str(value).strip() for value, _ in book.get_metadata("DC", key) if str(value).strip()),
                "",
            )
            return LoadedDocument(
                text="\n\n".join(parts),
                title=metadata("title"),
                author=metadata("creator"),
                language=metadata("language"),
                visuals=tuple(visuals),
                summary=BeautifulSoup(metadata("description"), "html.parser").get_text(" ", strip=True)[:6000],
                subjects=tuple(str(value).strip() for value, _ in book.get_metadata("DC", "subject"))[:30],
            )
        except DocumentLoadError:
            raise
        except Exception as exc:
            raise DocumentLoadError("L’EPUB est invalide ou endommagé.") from exc

    @staticmethod
    def _load_docx(content: bytes) -> LoadedDocument:
        try:
            from docx import Document

            document = Document(BytesIO(content))
            paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
            title = document.core_properties.title or ""
            author = document.core_properties.author or ""
            visuals = []
            # Extract embedded media directly from the DOCX package. Keep the
            # image bytes local; malformed media entries are skipped.
            from zipfile import ZipFile
            with ZipFile(BytesIO(content)) as archive:
                for name in archive.namelist():
                    if name.startswith("word/media/") and not name.endswith("/"):
                        media = Path(name).suffix.casefold()
                        media_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif"}.get(media, "application/octet-stream")
                        visuals.append(ExtractedVisual(archive.read(name), media_type, source_document=name, surrounding_text="\n\n".join(paragraphs[:3])[:1200]))
            return LoadedDocument("\n\n".join(paragraphs), title=title, author=author, visuals=tuple(visuals))
        except ImportError as exc:
            raise DocumentLoadError("L’import DOCX nécessite le paquet python-docx.") from exc
        except Exception as exc:
            raise DocumentLoadError("Le DOCX est invalide ou endommagé.") from exc


def detect_language(text: str) -> str:
    """Lightweight French/English detection for book metadata and UI defaults."""
    words = re.findall(r"[^\W\d_]+", text.casefold(), flags=re.UNICODE)
    sample = words[:12000]
    french = {"le", "la", "les", "des", "une", "est", "dans", "que", "qui", "pour", "avec", "mais", "elle", "il", "était", "sont", "aux", "ce", "pas"}
    english = {"the", "and", "is", "in", "that", "who", "for", "with", "but", "she", "he", "was", "are", "from", "this", "not", "his", "her", "they", "have"}
    fr_score = sum(word in french for word in sample) + sum(char in text.casefold() for char in "éèêàùçôî")
    en_score = sum(word in english for word in sample)
    if not sample or fr_score == en_score == 0:
        return ""
    return "fr" if fr_score >= en_score else "en"
