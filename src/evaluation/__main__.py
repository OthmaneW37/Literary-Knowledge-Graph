"""Command line: validate the annotated set or run it against the local engine.

    python -m evaluation validate
    python -m evaluation run [--category spoiler] [--only id1,id2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runner import EvaluationRunner
from .scenarios import CATEGORIES, validate_against_corpus

ROOT = Path.cwd()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evaluation", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["validate", "run"])
    parser.add_argument("--scenarios", type=Path, default=ROOT / "data/annotations/scenarios.json")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "data/library/evaluation")
    parser.add_argument("--category", choices=sorted(CATEGORIES))
    parser.add_argument("--only", help="Identifiants séparés par des virgules")
    args = parser.parse_args()

    from rag.engine import LiteraryAssistant
    assistant = LiteraryAssistant()
    runner = EvaluationRunner(lambda: assistant, args.scenarios, args.runs_dir,
                              args.scenarios.parent / "scenario_images")
    scenarios = runner.scenarios()
    if args.command == "validate":
        problems = validate_against_corpus(scenarios, assistant.index.passages, runner.images_dir)
        for problem in problems:
            print(f"✗ {problem}")
        print(f"{len(scenarios)} scénarios, {len(problems)} problème(s).")
        sys.exit(1 if problems else 0)

    selected = [s for s in scenarios if (not args.category or s.category == args.category)
                and (not args.only or s.id in args.only.split(","))]
    run = runner.run_all(selected)
    for row in run["results"]:
        mark = "✓" if row["passed"] else "✗"
        failed = "; ".join(c["detail"] for c in row["checks"] if not c["passed"]) or row.get("error") or ""
        print(f"{mark} {row['scenario_id']:<14} {row['outcome_label']:<45} {row['elapsed_ms']/1000:5.1f}s  {failed}")
    print(json.dumps(run["summary"], ensure_ascii=False, indent=2))
    print(f"Résultats enregistrés : {args.runs_dir / (run['run_id'] + '.json')}")


if __name__ == "__main__":
    main()
