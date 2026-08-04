from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub


def extract_text_from_epub(epub_path: str | Path) -> str:
    epub_path = Path(epub_path)
    book = epub.read_epub(str(epub_path))
    parts: list[str] = []

    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            if text:
                parts.append(text)

    return "\n\n".join(parts).strip()


def extract_epub_to_txt(epub_path: str | Path, txt_path: str | Path) -> Path:
    epub_path = Path(epub_path)
    txt_path = Path(txt_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    text = extract_text_from_epub(epub_path)
    txt_path.write_text(text, encoding="utf-8")
    return txt_path