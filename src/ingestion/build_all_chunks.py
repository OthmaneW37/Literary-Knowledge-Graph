from __future__ import annotations

import json
from pathlib import Path

from .chunk_text import build_chunks_for_work


def build_all_chunks(
    manifest_path: str | Path = "data/annotations/work_manifest.json",
    processed_dir: str | Path = "data/processed",
) -> list[str]:
    manifest_path = Path(manifest_path)
    processed_dir = Path(processed_dir)

    works = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = []

    for work in works:
        clean_path = processed_dir / f'{work["work_id"]}.clean.txt'
        output_path = processed_dir / f'{work["work_id"]}.chunks.json'
        build_chunks_for_work(work["work_id"], clean_path, output_path)
        outputs.append(str(output_path))

    return outputs


if __name__ == "__main__":
    print(json.dumps(build_all_chunks(), indent=2, ensure_ascii=False))
