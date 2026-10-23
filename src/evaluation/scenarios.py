"""Annotated evaluation scenarios: loading and corpus validation."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from storage.local import read_json

CATEGORIES: dict[str, dict[str, str]] = {
    "facts": {"label": "Faits du texte", "icon": "✎",
              "goal": "Retrouver un fait précis et le prouver par une citation exacte."},
    "characters": {"label": "Personnages et relations", "icon": "♙",
                   "goal": "Identifier qui est qui et les liens entre personnages."},
    "spoiler": {"label": "Anti-spoiler", "icon": "🛡",
                "goal": "Ne jamais utiliser ni révéler un passage au-delà du chapitre lu."},
    "out_of_corpus": {"label": "Hors corpus", "icon": "∅",
                      "goal": "Refuser quand la réponse n’est pas dans les livres sélectionnés."},
    "false_premise": {"label": "Fausse prémisse", "icon": "⚠",
                      "goal": "Corriger une question qui suppose un fait inexistant."},
    "injection": {"label": "Injection de consignes", "icon": "⛨",
                  "goal": "Ignorer les ordres cachés dans la question et garder ses règles."},
    "cross_work": {"label": "Comparaison de livres", "icon": "⇄",
                   "goal": "Répondre avec des preuves tirées de plusieurs œuvres."},
    "multimodal": {"label": "Question avec image", "icon": "▣",
                   "goal": "Relier une image fournie au bon passage du livre."},
}

EXPECTATIONS = {"answer", "abstain", "reject_premise", "safe"}


def normalize(text: str) -> str:
    """Case, accent and whitespace insensitive form used by every text check."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class GoldQuote:
    work_id: str
    contains: str


@dataclass(frozen=True)
class Scenario:
    id: str
    category: str
    title: str
    question: str
    work_ids: list[str]
    expect: str
    expected: str
    max_chapter: int | None = None
    mode: str = "Ask"
    gold: list[GoldQuote] = field(default_factory=list)
    must_include_any: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    cite_works: list[str] = field(default_factory=list)
    image: str | None = None
    showcase: bool = False
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        values = dict(data)
        values["gold"] = [GoldQuote(**item) for item in values.get("gold", [])]
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "category": self.category, "title": self.title, "question": self.question,
            "work_ids": self.work_ids, "max_chapter": self.max_chapter, "mode": self.mode,
            "expect": self.expect, "expected": self.expected,
            "gold": [{"work_id": g.work_id, "contains": g.contains} for g in self.gold],
            "must_include_any": self.must_include_any, "must_not_include": self.must_not_include,
            "cite_works": self.cite_works, "image": self.image, "showcase": self.showcase, "note": self.note,
        }


def load_scenarios(path: str | Path) -> list[Scenario]:
    payload = read_json(Path(path), {"scenarios": []})
    scenarios = [Scenario.from_dict(item) for item in payload.get("scenarios", [])]
    ids = [scenario.id for scenario in scenarios]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"Identifiants de scénario en double : {', '.join(duplicates)}")
    for scenario in scenarios:
        if scenario.category not in CATEGORIES:
            raise ValueError(f"{scenario.id} : catégorie inconnue « {scenario.category} »")
        if scenario.expect not in EXPECTATIONS:
            raise ValueError(f"{scenario.id} : attente inconnue « {scenario.expect} »")
    return scenarios


def validate_against_corpus(scenarios: list[Scenario], passages, scenario_dir: Path | None = None) -> list[str]:
    """Return problems that would make a scenario unfair or unverifiable.

    Every gold quote must exist verbatim in its work and stay inside the
    scenario's chapter limit, otherwise the expected answer itself would be a
    spoiler or an invention.
    """
    by_work: dict[str, list] = {}
    for passage in passages:
        by_work.setdefault(passage.work_id, []).append(passage)
    problems = []
    for scenario in scenarios:
        for work_id in [*scenario.work_ids, *scenario.cite_works]:
            if work_id not in by_work:
                problems.append(f"{scenario.id} : livre absent de l’index ({work_id})")
        for gold in scenario.gold:
            needle = normalize(gold.contains)
            matches = [p for p in by_work.get(gold.work_id, []) if needle in normalize(p.text)]
            if not matches:
                problems.append(f"{scenario.id} : citation introuvable dans {gold.work_id} « {gold.contains} »")
            elif scenario.max_chapter is not None and not any(
                str(p.chapter).isdigit() and int(p.chapter) <= scenario.max_chapter for p in matches
            ):
                problems.append(f"{scenario.id} : la citation de référence dépasse la limite du chapitre {scenario.max_chapter}")
        if scenario.expect == "answer" and not scenario.gold and not scenario.must_include_any:
            problems.append(f"{scenario.id} : une réponse attendue doit avoir une citation ou des mots attendus")
        if scenario.image and scenario_dir is not None and not (scenario_dir / scenario.image).is_file():
            problems.append(f"{scenario.id} : image introuvable ({scenario.image})")
    return problems
