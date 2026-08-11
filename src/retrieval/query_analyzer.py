# src/retrieval/query_analyzer.py

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import ollama


@dataclass
class QueryAnalysis:
    question_type: str = "textual"
    entities: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)

    use_lexical: bool = True
    use_graph: bool = False

    rewritten_query: str = ""

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        original_question: str,
    ) -> "QueryAnalysis":
        allowed_types = {
            "textual",
            "relationship",
            "character",
            "theme",
            "cross_work",
            "timeline",
        }

        question_type = str(
            payload.get("question_type", "textual")
        ).strip().lower()

        if question_type not in allowed_types:
            question_type = "textual"

        entities = payload.get("entities", [])
        if not isinstance(entities, list):
            entities = []

        themes = payload.get("themes", [])
        if not isinstance(themes, list):
            themes = []

        graph_types = {
            "relationship",
            "character",
            "cross_work",
            "timeline",
        }

        raw_use_lexical = payload.get("use_lexical", True)
        raw_use_graph = payload.get("use_graph", False)

        def as_bool(value: Any, default: bool) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().casefold()
                if normalized in {"true", "1", "yes", "oui"}:
                    return True
                if normalized in {"false", "0", "no", "non"}:
                    return False
            return default

        return cls(
            question_type=question_type,
            entities=[
                str(entity).strip()
                for entity in entities
                if str(entity).strip()
            ],
            themes=[
                str(theme).strip()
                for theme in themes
                if str(theme).strip()
            ],
            use_lexical=as_bool(raw_use_lexical, True),
            # Do not let an inconsistent model response disable the graph for
            # a question type that structurally requires it.
            use_graph=question_type in graph_types or as_bool(raw_use_graph, False),
            rewritten_query=(
                str(payload.get("rewritten_query", "")).strip()
                or original_question
            ),
        )


class QueryAnalyzer:
    """
    Analyse une question littéraire avant retrieval.

    Le LLM sert uniquement à classifier la question et identifier
    les personnages / thèmes mentionnés.

    Si Ollama est indisponible, un analyseur heuristique prend le relais.
    """

    GRAPH_HINTS = {
        "relation",
        "relations",
        "relationship",
        "relationships",
        "famille",
        "family",
        "père",
        "pere",
        "mère",
        "mere",
        "frère",
        "frere",
        "sœur",
        "soeur",
        "brother",
        "sister",
        "mother",
        "father",
        "between",
        "entre",
        "lié",
        "liée",
        "lie",
        "linked",
        "connecte",
        "connecté",
    }

    TIMELINE_HINTS = {
        "chronologie",
        "timeline",
        "avant",
        "après",
        "apres",
        "before",
        "after",
        "évolution",
        "evolution",
        "devient",
        "change",
    }

    THEME_HINTS = {
        "thème",
        "theme",
        "themes",
        "thèmes",
        "symbolise",
        "symbolism",
        "symbolique",
        "aliénation",
        "alienation",
        "culpabilité",
        "guilt",
    }

    ENTITY_STOP_WORDS = {
        "Affiche",
        "Analyse",
        "Comment",
        "Compare",
        "Explique",
        "Montre",
        "Pourquoi",
        "Quel",
        "Quelle",
        "Quels",
        "Quelles",
        "Qui",
        "Show",
        "What",
        "When",
        "Where",
        "Which",
        "Who",
        "Why",
    }

    def __init__(self, model: str = "qwen2.5:7b-instruct") -> None:
        self.model = model

    def analyze(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> QueryAnalysis:
        try:
            return self._analyze_with_llm(question, history)
        except Exception:
            return self._heuristic_analysis(question, history)

    def _analyze_with_llm(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> QueryAnalysis:

        recent_context = "\n".join(
            f"{item.get('role', '')}: {item.get('content', '')[:500]}"
            for item in (history or [])[-4:]
        )

        prompt = f"""
Tu analyses une question portant sur des romans.

Détermine quel type de recherche est nécessaire.

Types possibles :

- textual :
  question qui nécessite principalement de retrouver des passages du texte.

- relationship :
  question portant sur une relation entre personnages.

- character :
  question générale sur un personnage.

- theme :
  question portant sur un thème littéraire.

- cross_work :
  comparaison entre plusieurs œuvres.

- timeline :
  question portant sur une évolution ou une chronologie.

Retourne UNIQUEMENT un JSON valide :

{{
  "question_type": "textual",
  "entities": [],
  "themes": [],
  "use_lexical": true,
  "use_graph": false,
  "rewritten_query": "question/requête de recherche concise"
}}

Règles :

- use_lexical doit presque toujours être true.
- use_graph doit être true si les relations entre personnages,
  les personnages, une chronologie ou une comparaison structurée
  peuvent aider.
- entities contient uniquement les personnages explicitement nommés
  ou clairement identifiables grâce à l'historique.
- N'invente aucun personnage.
- themes contient les concepts littéraires explicitement demandés.
- rewritten_query doit préserver les noms propres.
- rewritten_query doit être une requête courte en anglais afin de rechercher
  dans le texte anglais, tout en conservant les noms propres.

Historique récent :
{recent_context or "(aucun)"}

Question :
{question}
""".strip()

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            format="json",
            options={
                "temperature": 0,
                "num_predict": 300,
            },
        )

        raw = str(response["message"]["content"]).strip()

        payload = json.loads(raw)

        return QueryAnalysis.from_dict(
            payload,
            original_question=question,
        )

    def _heuristic_analysis(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> QueryAnalysis:

        normalized = question.casefold()

        words = set(
            re.findall(
                r"[^\W_]+",
                normalized,
                flags=re.UNICODE,
            )
        )

        question_type = "textual"
        use_graph = False

        if words & self.GRAPH_HINTS:
            question_type = "relationship"
            use_graph = True

        elif words & self.TIMELINE_HINTS:
            question_type = "timeline"
            use_graph = True

        elif words & self.THEME_HINTS:
            question_type = "theme"

        entity_context = " ".join(
            [
                *(item.get("content", "") for item in (history or [])[-2:]),
                question,
            ]
        )
        candidates = re.findall(
            r"(?<![\w])(?:[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'’-]*|[A-Z]\.)(?:\s+(?:[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'’-]*|[A-Z]\.))*",
            entity_context,
        )
        entities: list[str] = []
        for candidate in candidates:
            cleaned = candidate.strip().rstrip(".")
            if cleaned in self.ENTITY_STOP_WORDS or not cleaned:
                continue
            if cleaned not in entities:
                entities.append(cleaned)

        return QueryAnalysis(
            question_type=question_type,
            entities=entities,
            themes=[],
            use_lexical=True,
            use_graph=use_graph,
            rewritten_query=question,
        )
