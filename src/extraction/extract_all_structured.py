from __future__ import annotations

import json
from pathlib import Path

from llm_extract import extract_chunk_llm


def extract_from_chunks_file(chunks_path: str | Path, output_path: str | Path) -> Path:
    chunks_path = Path(chunks_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    results = []

    for chunk in chunks:
        extraction = extract_chunk_llm(chunk["text"])
        results.append(
            {
                "work_id": chunk["work_id"],
                "chapter": chunk["chapter"],
                "chunk_id": chunk["chunk_id"],
                **extraction,
            }
        )

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    extract_from_chunks_file(
        chunks_path="data/processed/metamorphosis.chunks.json",
        output_path="data/processed/metamorphosis.extractions.json",
    )