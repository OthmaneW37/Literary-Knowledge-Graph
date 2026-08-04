from __future__ import annotations

import re
from pathlib import Path


START_MARKERS = [
    r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK .*? \*\*\*",
    r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .*? \*\*\*",
]

END_MARKERS = [
    r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK .*? \*\*\*",
    r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK .*? \*\*\*",
]


def _find_marker(text: str, patterns: list[str]) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.end()
    return None


def strip_gutenberg_header_footer(text: str) -> str:
    start = _find_marker(text, START_MARKERS)
    end = None
    for pattern in END_MARKERS:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            end = match.start()
            break

    if start is not None and end is not None and start < end:
        text = text[start:end]
    elif start is not None:
        text = text[start:]
    elif end is not None:
        text = text[:end]

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_quotes(text: str) -> str:
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "«": '"',
        "»": '"',
        "…": "...",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def clean_text(text: str) -> str:
    text = strip_gutenberg_header_footer(text)
    text = normalize_quotes(text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text.strip()