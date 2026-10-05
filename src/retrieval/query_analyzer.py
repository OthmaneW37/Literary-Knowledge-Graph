# src/retrieval/query_analyzer.py

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import ollama


# Frequent concepts in literary questions. Expanding both directions keeps
# lexical retrieval useful when the question and the edition use different
# languages, without paying for a second LLM call on every request.
BILINGUAL_TERMS = {
    "why": "pourquoi", "how": "comment", "who": "qui", "when": "quand",
    "father": "pere", "mother": "mere", "parents": "parents",
    "brother": "frere", "sister": "soeur", "son": "fils", "daughter": "fille",
    "family": "famille", "friend": "ami", "enemy": "ennemi",
    "love": "amour", "hate": "haine", "death": "mort", "murder": "meurtre",
    "kill": "tuer", "killed": "tue", "hide": "cacher", "hidden": "cache",
    "fear": "peur", "guilt": "culpabilite", "crime": "crime",
    "punishment": "chatiment", "justice": "justice", "freedom": "liberte",
    "power": "pouvoir", "authority": "autorite", "society": "societe",
    "loneliness": "solitude", "alienation": "alienation", "identity": "identite",
    "transformation": "transformation", "dream": "reve", "symbol": "symbole",
    "theme": "theme", "meaning": "signification", "relationship": "relation",
    "character": "personnage", "chapter": "chapitre", "scene": "scene",
    "beginning": "debut", "ending": "fin", "before": "avant", "after": "apres",
    "change": "changer", "become": "devenir", "think": "penser", "feel": "sentir",
    "say": "dire", "tell": "raconter", "leave": "quitter", "return": "retourner",
    "arrest": "arreter", "trial": "proces", "prison": "prison",
}


def _normalized_word(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in value if not unicodedata.combining(character))


def expand_bilingual_query(question: str) -> str:
    """Append useful French/English equivalents while preserving the question."""
    reverse = {french: english for english, french in BILINGUAL_TERMS.items()}
    translations: list[str] = []
    for raw_word in re.findall(r"[^\W_]+", question, flags=re.UNICODE):
        word = _normalized_word(raw_word)
        translated = BILINGUAL_TERMS.get(word) or reverse.get(word)
        if translated and translated not in translations:
            translations.append(translated)
    return " ".join([question, *translations]).strip()


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
        "Find",
        "Tell",
        "Describe",
        "Summarize",
        "Summarise",
        "Search",
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
        "How", "Is", "Are", "Does", "Do", "Can", "Could", "Explain",
        "I", "The", "In", "Give", "List", "Please", "Est", "Le", "La", "Les",
        "Je", "Résume", "Resume", "Donne", "Décris", "Liste", "Peux",
    }

    def __init__(
        self,
        model: str = "qwen3.5:4b-q4_K_M",
        mode: str | None = None,
        keep_alive: str = "15m",
        provider: Any | None = None,
    ) -> None:
        self.model = model
        self.mode = (mode or os.getenv("RAG_QUERY_ANALYSIS", "heuristic")).strip().casefold()
        self.keep_alive = keep_alive
        self.provider = provider

    def analyze(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> QueryAnalysis:
        if self.mode == "llm":
            try:
                return self._analyze_with_llm(question, history)
            except Exception:
                pass
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

        chat = self.provider.chat if self.provider else ollama.chat
        response = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            format="json",
            think=False,
            options={
                "temperature": 0,
                "num_ctx": 2048,
                "num_predict": 160,
            },
            keep_alive=self.keep_alive,
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

        pronouns = {"il", "elle", "ils", "elles", "lui", "he", "she", "they", "him", "her"}
        previous_questions = [item.get("content", "") for item in (history or [])[-6:] if item.get("role") == "user"]
        entity_context = " ".join([*previous_questions[-1:], question]) if words & pronouns else question
        candidates = re.findall(
            r"(?<![\w])(?:[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'’-]*|[A-Z]\.)(?:\s+(?:[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'’-]*|[A-Z]\.))*",
            entity_context,
        )
        entities: list[str] = []
        for candidate in candidates:
            cleaned = candidate.strip().rstrip(".")
            cleaned = re.sub(r"[’']s$", "", cleaned, flags=re.IGNORECASE)
            parts = cleaned.split()
            if parts and parts[0] in self.ENTITY_STOP_WORDS:
                cleaned = " ".join(parts[1:])
            if cleaned in self.ENTITY_STOP_WORDS or not cleaned:
                continue
            if cleaned not in entities:
                entities.append(cleaned)

        rewritten_query = expand_bilingual_query(question)
        pronouns = {"il", "elle", "ils", "elles", "lui", "he", "she", "they", "him", "her"}
        if words & pronouns and entities:
            rewritten_query = f"{rewritten_query} {' '.join(entities[-3:])}"

        return QueryAnalysis(
            question_type=question_type,
            entities=entities,
            themes=[],
            use_lexical=True,
            use_graph=use_graph,
            rewritten_query=rewritten_query,
        )
