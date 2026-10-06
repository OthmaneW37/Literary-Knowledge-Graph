from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def metrics(rankings, k=5):
    recall, ndcg, reciprocal = [], [], []
    for ranking, positives in rankings:
        ranking = list(dict.fromkeys(ranking))
        positive_set = set(positives)
        hit_ranks = [i + 1 for i, value in enumerate(ranking) if value in positive_set]
        recall.append(len(set(ranking[:k]) & positive_set) / max(len(positive_set), 1))
        dcg = sum(1 / math.log2(rank + 1) for rank in hit_ranks if rank <= k)
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(positive_set), k) + 1))
        ndcg.append(dcg / ideal if ideal else 0.0)
        first_hit = next((rank for rank in hit_ranks if rank <= 10), None)
        reciprocal.append(1 / first_hit if first_hit else 0.0)
    count = len(rankings)
    return {"examples": count, "recall_at_5": sum(recall) / count if count else None,
            "ndcg_at_5": sum(ndcg) / count if count else None, "mrr_at_10": sum(reciprocal) / count if count else None}


def main():
    parser = argparse.ArgumentParser(description="Score saved ranking runs against labeled passage IDs.")
    parser.add_argument("input", type=Path, help="JSONL rows with positive_ids and candidate_ids_by_system")
    parser.add_argument("--output", type=Path, default=Path("data/library/reranker-evaluation.json"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    systems = sorted({name for row in rows for name in row.get("candidate_ids_by_system", {})})
    results = {}
    for system in systems:
        rankings = [(row.get("candidate_ids_by_system", {}).get(system, []), row.get("positive_ids", [])) for row in rows]
        results[system] = metrics(rankings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results, "dataset_rows": len(rows)}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
