from __future__ import annotations

import json
import re
from pathlib import Path


def split_into_chapters(text: str) -> list[tuple[int, str]]:
    patterns = [
        r"(?im)^\s*chapter\s+([ivxlcdm0-9]+)\s*$",
        r"(?im)^\s*chapitre\s+([ivxlcdm0-9]+)\s*$",
    ]

    matches = []
    for pattern in patterns:
        matches.extend(list(re.finditer(pattern, text)))

    if not matches:
        return [(1, text.strip())]

    chapters = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapter_num = i + 1
        chapter_text = text[start:end].strip()
        chapters.append((chapter_num, chapter_text))

    return chapters


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)

    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

    return chunks


def build_chunks_for_work(
    work_id: str,
    clean_text_path: str | Path,
    output_path: str | Path,
) -> Path:
    clean_text_path = Path(clean_text_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text = clean_text_path.read_text(encoding="utf-8", errors="ignore")
    chapters = split_into_chapters(text)

    records = []
    for chapter_num, chapter_text in chapters:
        chunks = chunk_text(chapter_text)
        for idx, chunk in enumerate(chunks, start=1):
            records.append(
                {
                    "work_id": work_id,
                    "chapter": chapter_num,
                    "chunk_id": f"{work_id}_ch{chapter_num:02d}_p{idx:03d}",
                    "text": chunk,
                }
            )

    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    build_chunks_for_work(
        work_id="metamorphosis",
        clean_text_path="data/processed/metamorphosis.clean.txt",
        output_path="data/processed/metamorphosis.chunks.json",
    )