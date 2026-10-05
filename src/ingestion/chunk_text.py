from __future__ import annotations

import json
import re
from pathlib import Path


def split_into_chapters(text: str) -> list[tuple[int, str]]:
    """Recognize headings without throwing away short chapters or epilogues."""
    heading = re.compile(
        r"(?im)^[ \t]*(?:#{1,4}[ \t]+)?(?:"
        r"(?:chapter|chapitre|part|partie)\s+(?:[ivxlcdm\d]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|premier|première)\b[^\n]*"
        r"|[IVXLCDM]{1,10})[ \t]*$"
    )
    matches = list(heading.finditer(text))
    if not matches:
        return [(1, text.strip())] if text.strip() else []
    chapters = []
    preface = text[:matches[0].start()].strip()
    if len(preface.split()) >= 20:
        chapters.append((0, preface))
    number = 0
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        duplicate = any(other.group().strip().casefold() == match.group().strip().casefold() for other in matches[position + 1:])
        if duplicate and len(body.split()) < 50:
            continue
        number += 1
        chapters.append((number, text[match.start():end].strip()))
    return chapters


def _chunk_text_with_offsets(text: str, chunk_size: int = 600, overlap: int = 100) -> list[tuple[str, int, int]]:
    """Bounded word windows ending at paragraph boundaries whenever possible."""
    if chunk_size < 1 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size must be positive and 0 <= overlap < chunk_size")
    words = list(re.finditer(r"\S+", text))
    chunks = []
    first = 0
    while first < len(words):
        stop = min(first + chunk_size, len(words))
        if stop < len(words):
            boundaries = [
                index for index in range(first + max(1, chunk_size // 2), stop + 1)
                if index < len(words) and re.search(r"\n[ \t]*\n", text[words[index - 1].end():words[index].start()])
            ]
            if boundaries:
                stop = boundaries[-1]
        start_char, end_char = words[first].start(), words[stop - 1].end()
        chunks.append((text[start_char:end_char], start_char, end_char))
        if stop == len(words):
            break
        first = max(first + 1, stop - overlap)
    return chunks


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    return [chunk for chunk, _, _ in _chunk_text_with_offsets(text, chunk_size, overlap)]


def build_chunks_for_work(
    work_id: str, clean_text_path: str | Path, output_path: str | Path,
    chunk_size: int = 600, overlap: int = 100,
) -> Path:
    """Offsets refer to cleaned text with internal PDF page markers removed."""
    source = Path(clean_text_path).read_text(encoding="utf-8")
    pages = []
    pieces = []
    cursor = 0
    length = 0
    for marker in re.finditer(r"(?m)^\[\[PAGE:(\d+)\]\][ \t]*\n?", source):
        piece = source[cursor:marker.start()]
        pieces.append(piece)
        length += len(piece)
        pages.append((length, int(marker.group(1))))
        cursor = marker.end()
    pieces.append(source[cursor:])
    text = "".join(pieces)
    records = []
    cursor = 0
    for chapter, body in split_into_chapters(text):
        chapter_start = text.index(body, cursor)
        cursor = chapter_start + len(body)
        for index, (chunk, start, end) in enumerate(_chunk_text_with_offsets(body, chunk_size, overlap), 1):
            start += chapter_start
            end += chapter_start
            active = [page for position, page in pages if position <= start]
            page = active[-1] if active else None
            covered = list(dict.fromkeys(([page] if page else []) + [p for position, p in pages if start < position < end]))
            records.append({
                "work_id": work_id, "chapter": chapter,
                "section": body.splitlines()[0][:120] if len(body.splitlines()) > 1 else "",
                "page": page, "pages": covered, "chunk_index": index,
                "start_char": start, "end_char": end,
                "chunk_id": f"{work_id}_ch{chapter:02d}_p{index:03d}", "text": chunk,
            })
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
