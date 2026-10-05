from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from bs4 import BeautifulSoup


class DocumentLoadError(ValueError):
    """A document cannot be converted into useful, searchable text."""


@dataclass(frozen=True)
class LoadedDocument:
    text: str
    title: str = ""
    author: str = ""
    language: str = ""
    page_count: int = 0
    likely_scanned: bool = False


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
            page_text_lengths = []
            for number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                page_text_lengths.append(len(text.strip()))
                if text.strip():
                    sections.append(f"[[PAGE:{number}]]\n{text.strip()}")
        except Exception as exc:
            raise DocumentLoadError("Le PDF est invalide ou endommagé.") from exc
        text = "\n\n".join(sections)
        likely_scanned = not text.strip() or (
            len(text.strip()) < DocumentLoader.MIN_PDF_TEXT_CHARS
            or (len(page_text_lengths) >= 2 and sum(page_text_lengths) / len(page_text_lengths) < 35)
        )
        return LoadedDocument(text, page_count=len(page_text_lengths), likely_scanned=likely_scanned)

    @staticmethod
    def _load_epub(content: bytes) -> LoadedDocument:
        from ebooklib import ITEM_DOCUMENT, epub

        try:
            with BytesIO(content) as archive:
                book = epub.read_epub(archive)
            parts = []
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
            metadata = lambda key: next(
                (str(value).strip() for value, _ in book.get_metadata("DC", key) if str(value).strip()),
                "",
            )
            return LoadedDocument(
                text="\n\n".join(parts),
                title=metadata("title"),
                author=metadata("creator"),
                language=metadata("language"),
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
            return LoadedDocument("\n\n".join(paragraphs), title=title, author=author)
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
