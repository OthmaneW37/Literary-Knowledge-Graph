from __future__ import annotations

import json
from pathlib import Path

from clean_text import clean_text
from extract_epub import extract_epub_to_txt


def process_corpus(
    manifest_path: str | Path = "data/annotations/work_manifest.json",
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
) -> list[dict]:
    manifest_path = Path(manifest_path)
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    works = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []

    for work in works:
        raw_path = raw_dir / work["raw_filename"]
        extracted_path = processed_dir / f'{work["work_id"]}.extracted.txt'
        cleaned_path = processed_dir / f'{work["work_id"]}.clean.txt'

        extract_epub_to_txt(raw_path, extracted_path)
        cleaned = clean_text(extracted_path.read_text(encoding="utf-8", errors="ignore"))
        cleaned_path.write_text(cleaned, encoding="utf-8")

        results.append(
            {
                "work_id": work["work_id"],
                "title": work["title"],
                "raw_path": str(raw_path),
                "extracted_path": str(extracted_path),
                "clean_path": str(cleaned_path),
            }
        )

    return results


if __name__ == "__main__":
    print(json.dumps(process_corpus(), indent=2, ensure_ascii=False))