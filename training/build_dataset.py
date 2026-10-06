from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_records(records):
    normalized = []
    for number, row in enumerate(records, 1):
        question = str(row.get("question", "")).strip()
        positive = str(row.get("positive", "")).strip()
        negatives = row.get("hard_negatives", [])
        if not question or not positive or not isinstance(negatives, list) or not negatives:
            raise ValueError(f"Ligne {number}: question, positive et hard_negatives non vides sont requis.")
        cleaned = list(dict.fromkeys(str(item).strip() for item in negatives if str(item).strip()))
        if not cleaned or positive in cleaned:
            raise ValueError(f"Ligne {number}: négatifs vides ou identiques au passage positif.")
        normalized.append({"question": question, "positive": positive, "hard_negatives": cleaned,
                           **{key: row[key] for key in ("work_id", "document_id", "group_id", "source_ids", "reviewed_by") if key in row}})
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Validate and normalize literary reranker examples (JSONL).")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    normalized = validate_records(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in normalized), encoding="utf-8")
    print(f"Validated {len(normalized)} real examples -> {args.output}")


if __name__ == "__main__":
    main()
