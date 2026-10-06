from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from time import perf_counter
import math

from .engine import LiteraryAssistant
from storage.local import read_json, write_json


def evaluate(questions_path: str | Path = "data/annotations/eval_questions.json",
             top_k: int = 6, generate_answers: bool = False,
             assistant: LiteraryAssistant | None = None, lexical_only: bool = False) -> dict:
    path = Path(questions_path)
    cases = read_json(path, [])
    references = read_json(path.with_name("gold_passages.json"), {})
    assistant = assistant or LiteraryAssistant()
    if lexical_only:
        assistant.hybrid_retriever.semantic = None
    normalize = lambda value: re.sub(r"\s+", " ", value).casefold().strip()
    rows = []
    for case in cases:
        expected = set(case.get("expected_works", []))
        if case.get("expected_work"):
            expected.add(case["expected_work"])
        started = perf_counter()
        retrieval = assistant.hybrid_retriever.retrieve(case["question"], top_k=top_k)
        passages = retrieval.passages
        actual_works = {p.work_id for p in passages}
        row = {
            "id": case["id"], "question": case["question"],
            "retrieved_ids": [p.chunk_id for p in passages],
            "retrieval_ms": round((perf_counter() - started) * 1000),
        }
        if case.get("expect_refusal"):
            row["retrieval_refusal_correct"] = not passages and not retrieval.graph.relationships
        else:
            row.update({
                "retrieval_hit": bool(expected & actual_works),
                "work_recall": len(expected & actual_works) / max(len(expected), 1),
                "work_precision": sum(p.work_id in expected for p in passages) / max(len(passages), 1),
            })
        gold = references.get(case["id"], [])
        if gold:
            # Quotes are stable anchors even after changing chunk boundaries.
            expected_ids = {
                p.chunk_id for p in assistant.index.passages
                if any(p.work_id == ref["work_id"] and normalize(ref["contains"]) in normalize(p.text) for ref in gold)
            }
            hits = len(expected_ids & {p.chunk_id for p in passages})
            row.update({
                "gold_available": bool(expected_ids), "passage_hit": hits > 0,
                "annotated_passage_recall": hits / max(len(expected_ids), 1),
                "annotated_source_precision": hits / max(len(passages), 1),
            })
            ranks = [i + 1 for i, passage in enumerate(passages) if passage.chunk_id in expected_ids]
            ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(5, len(expected_ids)) + 1))
            row.update({'recall_at_5': len(expected_ids & {p.chunk_id for p in passages[:5]}) / max(1, len(expected_ids)),
                        'ndcg_at_5': sum(1 / math.log2(rank + 1) for rank in ranks if rank <= 5) / ideal if ideal else 0,
                        'mrr_at_10': next((1 / rank for rank in ranks if rank <= 10), 0)})
        if generate_answers:
            answer = assistant.answer(case["question"], top_k=top_k)
            row.update({
                "answer": answer.text, "has_citations": bool(answer.citations),
                "answer_used_model": answer.used_model,
                "citation_ids_valid": bool(answer.citations) and all(assistant.index.get_passage(p.chunk_id) for p in answer.citations),
                "generation_validation_failed": bool(answer.raw.get("error")),
                "answer_ms": answer.retrieval_ms + answer.generation_ms,
            })
            if case.get("expect_refusal"):
                row["answer_refusal_correct"] = not answer.citations and not answer.raw.get("error")
        rows.append(row)

    def mean(key):
        values = [float(row[key]) for row in rows if key in row]
        return round(sum(values) / len(values), 3) if values else None

    def percentile(key, quantile):
        values = sorted(float(row[key]) for row in rows if key in row)
        return values[max(0, math.ceil(len(values) * quantile) - 1)] if values else None

    return {
        "question_count": len(rows), "retrieval_hit_rate": mean("retrieval_hit"),
        "mean_work_recall": mean("work_recall"), "mean_work_precision": mean("work_precision"),
        "passage_hit_rate": mean("passage_hit"),
        "annotated_source_precision": mean("annotated_source_precision"),
        "refusal_accuracy": mean("retrieval_refusal_correct"),
        "generation_validation_failure_rate": mean("generation_validation_failed"),
        "mean_retrieval_ms": mean("retrieval_ms"), "mean_answer_ms": mean("answer_ms"),
        "recall_at_5": mean('recall_at_5'), "ndcg_at_5": mean('ndcg_at_5'), "mrr_at_10": mean('mrr_at_10'),
        "retrieval_p50_ms": percentile('retrieval_ms', .5), "retrieval_p95_ms": percentile('retrieval_ms', .95),
        "answer_p50_ms": percentile('answer_ms', .5), "answer_p95_ms": percentile('answer_ms', .95),
        "retrieval_mode": 'lexical_only' if lexical_only else 'hybrid', "top_k": top_k,
        "note": "Gold quotes are a small, non-exhaustive annotation set. Citation provenance is checked; factual entailment still requires human review.",
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local retrieval, evidence provenance, refusals and latency.")
    parser.add_argument("--questions", type=Path, default=Path("data/annotations/eval_questions.json"))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--lexical-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.questions, args.top_k, args.generate, lexical_only=args.lexical_only)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
