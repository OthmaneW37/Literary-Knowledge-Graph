"""Deterministic, explainable verdicts for one scenario run.

No model judges another model here: every check reads structured output
(status, citations, chapters) or exact text, so a jury can audit each verdict.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ingestion.loaders import detect_language
from rag.engine import LiteraryAssistant
from retrieval.spoiler_policy import SpoilerPolicy

from .scenarios import Scenario, normalize

OUTCOME_LABELS = {
    "answered": "Réponse rédigée et vérifiée",
    "quoted": "Citation exacte, sans reformulation",
    "fallback": "Pas de réponse vérifiée (passages consultés)",
    "premise_rejected": "Prémisse corrigée",
    "spoiler_guarded": "Suite protégée (anti-spoiler)",
    "abstained": "Abstention faute de preuve",
}


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def classify(answer, status: str | None = None) -> str:
    raw = getattr(answer, "raw", {}) or {}
    if raw.get("spoiler_guard"):
        return "spoiler_guarded"
    if raw.get("premise_rejected"):
        return "premise_rejected"
    if raw.get("quoted"):
        return "quoted"
    if not answer.citations:
        return "abstained"
    if not answer.used_model:
        return "fallback"
    return "answered"


def _contains(text: str, terms: list[str]) -> list[str]:
    """Whole-word matches, so "dies" never matches "studies"."""
    haystack = normalize(text)
    return [term for term in terms
            if re.search(rf"(?<!\w){re.escape(normalize(term))}(?!\w)", haystack)]


def evaluate_checks(scenario: Scenario, answer, outcome: str) -> list[Check]:
    checks: list[Check] = []
    citations = list(answer.citations)
    label = OUTCOME_LABELS[outcome]

    if scenario.expect == "answer":
        answered = outcome in {"answered", "quoted"}
        checks.append(Check("Réponse attendue", answered,
                            f"Résultat : {label.lower()}." if not answered
                            else "Réponse fondée sur des citations vérifiées mot pour mot."))
    elif scenario.expect == "abstain":
        ok = outcome in {"abstained", "spoiler_guarded", "premise_rejected"}
        checks.append(Check("Refus ou abstention", ok,
                            "Le moteur n’a affirmé aucun fait sans preuve." if ok
                            else f"Le moteur aurait dû s’abstenir, résultat : {label.lower()}."))
    elif scenario.expect == "reject_premise":
        ok = outcome in {"premise_rejected", "abstained"}
        checks.append(Check("Prémisse non validée", ok,
                            "La question repose sur un fait faux et le moteur ne l’a pas confirmé." if ok
                            else f"Le moteur n’a pas signalé la fausse prémisse, résultat : {label.lower()}."))

    if scenario.gold and scenario.expect == "answer":
        quotes = {(gold.work_id, normalize(gold.contains)) for gold in scenario.gold}
        hit = next((p for p in citations for work_id, quote in quotes
                    if p.work_id == work_id and quote in normalize(p.text)), None)
        checks.append(Check("Passage de référence cité", hit is not None,
                            f"Citation trouvée dans {hit.chunk_id}." if hit
                            else "Aucune citation ne contient le passage de référence annoté."))

    if scenario.must_include_any and scenario.expect == "answer":
        found = _contains(answer.text, scenario.must_include_any)
        checks.append(Check("Contenu attendu", bool(found),
                            f"Contient « {found[0]} »." if found
                            else f"Aucun des termes attendus : {', '.join(scenario.must_include_any)}."))

    if scenario.must_not_include:
        found = _contains(answer.text, scenario.must_not_include)
        checks.append(Check("Aucun contenu interdit", not found,
                            "Aucun terme interdit dans la réponse." if not found
                            else f"La réponse contient « {found[0]} »."))

    if scenario.max_chapter is not None:
        late = [p for p in citations if not SpoilerPolicy.allows(p.chapter, scenario.max_chapter)]
        checks.append(Check("Limite de chapitre respectée", not late,
                            f"Toutes les preuves viennent des chapitres 1 à {scenario.max_chapter}." if not late
                            else f"{late[0].chunk_id} vient du chapitre {late[0].chapter}."))

    foreign = [p for p in citations if p.work_id not in scenario.work_ids]
    if citations:
        checks.append(Check("Livres sélectionnés uniquement", not foreign,
                            "Les citations viennent uniquement des livres choisis." if not foreign
                            else f"{foreign[0].chunk_id} vient d’un livre non sélectionné."))

    if scenario.cite_works and outcome == "answered":
        missing = [work for work in scenario.cite_works if work not in {p.work_id for p in citations}]
        checks.append(Check("Preuves dans chaque livre", not missing,
                            "Chaque livre comparé est cité." if not missing
                            else f"Aucune citation de : {', '.join(missing)}."))

    if outcome == "answered":
        # The engine decides the answer language from the question; reuse its rule.
        asked = {"French": "fr", "English": "en"}[LiteraryAssistant._answer_language(scenario.question)]
        replied = detect_language(answer.text.split("[")[0])
        same = asked == replied or not replied
        checks.append(Check("Langue de la question", same,
                            "Réponse dans la langue de la question." if same
                            else f"Question en « {asked} », réponse en « {replied} »."))
    return checks
