"""Run scenarios through the same workflow as the chat and keep a run history."""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from security.budget import configured_budget, current_budget
from storage.local import read_json, write_json
from workflows.narrative_graph import NarrativeWorkflow

from .checks import OUTCOME_LABELS, classify, evaluate_checks
from .scenarios import CATEGORIES, Scenario, load_scenarios


def _passage(passage) -> dict:
    return {"work_id": passage.work_id, "work_title": passage.work_title, "chapter": passage.chapter,
            "chunk_id": passage.chunk_id, "text": passage.text, "page": passage.page,
            "citation_label": passage.citation_label}


def summarize(results: list[dict]) -> dict:
    by_category = {}
    for category in CATEGORIES:
        rows = [row for row in results if row["category"] == category]
        if rows:
            by_category[category] = {"passed": sum(row["passed"] for row in rows), "total": len(rows)}
    timed = sorted(row["elapsed_ms"] for row in results if not row.get("error"))
    return {
        "total": len(results), "passed": sum(row["passed"] for row in results),
        "errors": sum(bool(row.get("error")) for row in results),
        "by_category": by_category,
        "median_ms": timed[len(timed) // 2] if timed else None,
    }


class EvaluationRunner:
    def __init__(self, assistant_factory, scenarios_path: Path, runs_dir: Path, images_dir: Path,
                 vision_factory=None) -> None:
        self.assistant_factory = assistant_factory
        self.scenarios_path = Path(scenarios_path)
        self.runs_dir = Path(runs_dir)
        self.images_dir = Path(images_dir)
        self.vision_factory = vision_factory
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self.progress: dict = {"state": "idle"}

    def scenarios(self) -> list[Scenario]:
        return load_scenarios(self.scenarios_path)

    def scenario(self, scenario_id: str) -> Scenario:
        found = next((item for item in self.scenarios() if item.id == scenario_id), None)
        if found is None:
            raise KeyError(scenario_id)
        return found

    def run_scenario(self, scenario: Scenario) -> dict:
        assistant = self.assistant_factory()
        state = {"user_id": "evaluation", "question": scenario.question, "work_ids": scenario.work_ids,
                 "top_k": 4, "history": [], "max_chapter": scenario.max_chapter, "mode": scenario.mode}
        observation = None
        started = perf_counter()
        token = current_budget.set(configured_budget())
        try:
            if scenario.image:
                observation = self._observe(scenario)
                if observation:
                    terms = [observation.ocr_text, observation.visual_description,
                             *observation.visible_entities, *observation.objects]
                    state["retrieval_query"] = " ".join([scenario.question, *(t for t in terms if t)])
                    state["image_analysis"] = observation.model_dump()
            outcome_state = NarrativeWorkflow(assistant).invoke(state)
            answer = outcome_state["answer"]
            outcome = classify(answer)
            checks = evaluate_checks(scenario, answer, outcome)
            result = {
                "answer": answer.text, "status": outcome_state.get("status", "answered"),
                "citations": [_passage(p) for p in answer.citations],
                "used_model": answer.used_model, "error": None,
                "visual_observation": observation.model_dump() if observation else None,
            }
        except Exception as exc:  # A failing scenario must not stop a full run.
            logging.exception("Scenario %s failed", scenario.id)
            outcome, checks = "abstained", []
            result = {"answer": "", "status": "error", "citations": [], "used_model": False,
                      "error": str(exc) or exc.__class__.__name__, "visual_observation": None}
        finally:
            current_budget.reset(token)
        return {
            "scenario_id": scenario.id, "category": scenario.category, "question": scenario.question,
            "outcome": outcome, "outcome_label": OUTCOME_LABELS[outcome],
            "passed": bool(checks) and all(check.passed for check in checks) and not result["error"],
            "checks": [check.to_dict() for check in checks],
            "elapsed_ms": round((perf_counter() - started) * 1000),
            "model": getattr(assistant, "model", ""),
            "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **result,
        }

    def _observe(self, scenario: Scenario):
        try:
            from vision import VisionAnalyzer
            analyzer = self.vision_factory() if self.vision_factory else VisionAnalyzer()
            return analyzer.analyze(self.images_dir / scenario.image, scenario.question)
        except Exception as exc:
            logging.info("Vision unavailable for scenario %s: %s", scenario.id, exc)
            return None

    # Full runs -----------------------------------------------------------
    def start(self, scenario_ids: list[str] | None = None) -> dict:
        with self._lock:
            if self.progress.get("state") == "running":
                raise RuntimeError("Une évaluation complète est déjà en cours.")
            scenarios = [s for s in self.scenarios() if not scenario_ids or s.id in scenario_ids]
            if not scenarios:
                raise ValueError("Aucun scénario à exécuter.")
            self._cancel.clear()
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6]
            self.progress = {"state": "running", "run_id": run_id, "done": 0, "total": len(scenarios),
                             "passed": 0, "current": scenarios[0].id, "results": []}
        threading.Thread(target=self._run_all, args=(run_id, scenarios), daemon=True).start()
        return self.status()

    def cancel(self) -> dict:
        self._cancel.set()
        return self.status()

    def status(self) -> dict:
        return {key: value for key, value in self.progress.items() if key != "results"}

    def run_all(self, scenarios: list[Scenario] | None = None) -> dict:
        """Synchronous variant used by the command line."""
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6]
        scenarios = scenarios or self.scenarios()
        self.progress = {"state": "running", "run_id": run_id, "done": 0, "total": len(scenarios),
                         "passed": 0, "current": scenarios[0].id, "results": []}
        return self._run_all(run_id, scenarios)

    def _run_all(self, run_id: str, scenarios: list[Scenario]) -> dict:
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        results = []
        state = "done"
        try:
            for scenario in scenarios:
                if self._cancel.is_set():
                    state = "cancelled"
                    break
                self.progress["current"] = scenario.id
                row = self.run_scenario(scenario)
                results.append(row)
                self.progress.update(done=len(results), passed=sum(r["passed"] for r in results))
        except Exception as exc:
            logging.exception("Evaluation run failed")
            state, self.progress["error"] = "failed", str(exc)
        run = {
            "run_id": run_id, "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "state": state, "model": results[0]["model"] if results else "",
            "scenario_count": len(scenarios), "summary": summarize(results), "results": results,
        }
        if results:
            write_json(self.runs_dir / f"{run_id}.json", run)
        self.progress.update(state=state, current=None)
        return run

    # History -------------------------------------------------------------
    def runs(self) -> list[dict]:
        rows = []
        for path in sorted(self.runs_dir.glob("*.json"), reverse=True):
            try:
                run = read_json(path, None)
            except ValueError:
                continue
            if run:
                rows.append({key: run.get(key) for key in
                             ("run_id", "started_at", "finished_at", "state", "model", "scenario_count", "summary")})
        return rows

    def run(self, run_id: str) -> dict | None:
        if not run_id.replace("-", "").isalnum():
            return None
        return read_json(self.runs_dir / f"{run_id}.json", None)

    def latest_results(self) -> dict[str, dict]:
        """Most recent result per scenario across full and single runs."""
        latest: dict[str, dict] = {}
        paths = [*self.runs_dir.glob("*.json"), *(self.runs_dir / "single").glob("*.json")]
        for path in paths:
            try:
                run = read_json(path, None) or {}
            except ValueError:
                continue
            for row in run.get("results", []):
                current = latest.get(row["scenario_id"])
                if current is None or row.get("ran_at", "") >= current.get("ran_at", ""):
                    latest[row["scenario_id"]] = row
        return latest

    def record_single(self, row: dict) -> None:
        """Keep single demo runs so the list shows the last verdict of each scenario."""
        path = self.runs_dir / "single" / f"{row['scenario_id']}.json"
        write_json(path, {"run_id": "single", "results": [row]})
