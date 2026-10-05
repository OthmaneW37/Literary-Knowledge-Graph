from __future__ import annotations

import json
import os
import re
from typing import Any

import ollama
from dotenv import load_dotenv


load_dotenv()
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3.5:4b-q4_K_M")

MAX_MENTION_WORDS = 4
MAX_CONTEXT_CHARS = 300
MAX_EVIDENCE_CHARS = 500

PRONOUNS = {
    "he",
    "she",
    "it",
    "they",
    "we",
    "i",
    "you",
    "him",
    "her",
    "them",
    "his",
    "their",
    "our",
    "your",
    "its",
}

ARTICLES_AND_DETERMINERS = {
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "his",
    "her",
    "their",
    "its",
}

OBJECT_AND_CONCEPT_WORDS = {
    "clock",
    "door",
    "room",
    "pain",
    "silence",
    "weather",
    "train",
    "picture",
    "chair",
    "muff",
    "frame",
    "table",
    "window",
    "bed",
    "wall",
    "needle",
    "lamp",
    "sound",
    "voice",
    "conversation",
    "budget",
    "time",
    "itch",
    "legs",
    "gas",
    "gaslight",
    "couch",
    "newspaper",
    "alarm",
    "locksmith",
    "weather",
    "stubbornness",
    "discourtesy",
    "dizziness",
    "silence",
    "family",
    "work",
    "room",
    "picture",
    "sample",
    "samples",
    "drawer",
    "drawers",
    "legs",
    "household",
}

VERB_WORDS = {
    "would",
    "could",
    "should",
    "was",
    "were",
    "is",
    "are",
    "be",
    "being",
    "been",
    "run",
    "runs",
    "leave",
    "leaves",
    "left",
    "keep",
    "keeps",
    "kept",
    "speak",
    "speaks",
    "spoke",
    "tug",
    "tugs",
    "tugged",
    "throw",
    "throws",
    "threw",
    "sit",
    "sits",
    "sat",
    "sleep",
    "sleeps",
    "slept",
    "learn",
    "learns",
    "learned",
    "go",
    "goes",
    "went",
    "come",
    "comes",
    "came",
    "see",
    "sees",
    "saw",
    "hear",
    "hears",
    "heard",
    "exclaim",
    "exclaims",
    "exclaimed",
    "ask",
    "asks",
    "asked",
    "say",
    "says",
    "said",
    "cry",
    "cries",
    "cried",
    "work",
    "works",
    "worked",
    "rent",
    "rents",
    "rented",
}

ALLOWED_RELATION_TYPES = {
    "FAMILY",
    "AUTHORITY_OVER",
    "CONFLICT_WITH",
    "ASSISTS",
    "COMMUNICATES_WITH",
    "OBSERVES",
}


def build_prompt(chunk_text: str) -> str:
    return f"""
You extract information from a literary passage.

Return ONLY valid JSON. Do not use Markdown and do not add explanations.

Use exactly this structure:
{{
  "mentions": [
    {{
      "text": "short person name or stable human role exactly as written",
      "context": "short context from the passage",
      "chapter": ""
    }}
  ],
  "relations": [
    {{
      "source": "exact person mention",
      "target": "exact person mention",
      "type": "FAMILY",
      "evidence": "short exact quote"
    }}
  ],
  "themes": [
    {{
      "name": "theme explicitly supported by the passage",
      "evidence": "short exact quote"
    }}
  ],
  "events": [
    {{
      "summary": "short event summary",
      "participants": ["exact person mention"],
      "evidence": "short exact quote"
    }}
  ],
  "evidence_quote": "short exact quote",
  "confidence": 0.0
}}

Extraction rules:
- Extract only people, named characters, or short stable human roles.
- A valid mention can be a name such as "Gregor Samsa" or a role such as "chief clerk".
- Do not extract pronouns: he, she, it, they, his, her, their, them.
- Do not extract objects, rooms, places, body parts, feelings, abstract concepts, actions, or events.
- Do not extract complete sentences or long descriptions.
- Do not invent a full name that does not appear in the passage.
- Do not resolve coreference. Preserve the exact surface form.
- Extract only themes that are genuinely supported by the passage.
- If there are no valid items, return empty lists.
- Keep every mention short, preferably between one and four words.
- The confidence value must be between 0 and 1.

Passage:
{chunk_text}
"""


def clean_json(text: str) -> str:
    text = str(text).strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return ""

    text = text[start : end + 1]

    # Supprime les virgules finales avant } ou ].
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return text


def normalize_string(value: Any, max_chars: int | None = None) -> str:
    if not isinstance(value, str):
        return ""

    value = value.strip()
    value = re.sub(r"\s+", " ", value)

    if max_chars is not None:
        value = value[:max_chars]

    return value


def normalize_words(text: str) -> list[str]:
    cleaned = text.lower()
    cleaned = cleaned.replace("’", "'")
    cleaned = re.sub(r"[^\w\s'-]", " ", cleaned)
    return [word for word in cleaned.split() if word]


def is_probable_sentence(text: str) -> bool:
    words = normalize_words(text)

    if len(words) > MAX_MENTION_WORDS:
        return True

    if any(word in VERB_WORDS for word in words):
        return True

    if re.search(r"[.!?]{1,}", text):
        return True

    return False


def is_valid_mention(text: Any) -> bool:
    if not isinstance(text, str):
        return False

    mention = normalize_string(text)

    if not mention:
        return False

    words = normalize_words(mention)

    if not words:
        return False

    if len(words) > MAX_MENTION_WORDS:
        return False

    if mention.startswith(("\"", "'")) or mention.endswith(("\"", "'")):
        return False

    if any(word in PRONOUNS for word in words):
        return False

    # Rejette "the chief clerk", "his sister", etc.
    # Le LLM doit retourner "chief clerk" ou "sister".
    if words[0] in ARTICLES_AND_DETERMINERS:
        return False

    # Utilise des mots entiers, pas des sous-chaînes.
    if any(word in OBJECT_AND_CONCEPT_WORDS for word in words):
        return False

    if is_probable_sentence(mention):
        return False

    # Rejette les questions ou phrases manifestes.
    if "?" in mention or "!" in mention:
        return False

    return True


def normalize_mention(item: Any) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None

    text = normalize_string(item.get("text"))

    if not is_valid_mention(text):
        return None

    context = normalize_string(
        item.get("context"),
        max_chars=MAX_CONTEXT_CHARS,
    )

    chapter = normalize_string(item.get("chapter"), max_chars=100)

    return {
        "text": text,
        "context": context,
        "chapter": chapter,
    }


def normalize_relations(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []

    relations = []

    for item in items:
        if not isinstance(item, dict):
            continue

        source = normalize_string(item.get("source"))
        target = normalize_string(item.get("target"))
        relation_type = normalize_string(item.get("type")).upper()
        evidence = normalize_string(
            item.get("evidence"),
            max_chars=MAX_EVIDENCE_CHARS,
        )

        if not source or not target:
            continue

        if relation_type not in ALLOWED_RELATION_TYPES:
            continue

        if not is_valid_mention(source) or not is_valid_mention(target):
            continue

        relations.append(
            {
                "source": source,
                "target": target,
                "type": relation_type,
                "evidence": evidence,
            }
        )

    return relations


def normalize_themes(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []

    themes = []

    for item in items:
        if isinstance(item, str):
            name = normalize_string(item)
            evidence = ""
        elif isinstance(item, dict):
            name = normalize_string(item.get("name"))
            evidence = normalize_string(
                item.get("evidence"),
                max_chars=MAX_EVIDENCE_CHARS,
            )
        else:
            continue

        if not name:
            continue

        themes.append(
            {
                "name": name,
                "evidence": evidence,
            }
        )

    return themes


def normalize_events(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []

    events = []

    for item in items:
        if not isinstance(item, dict):
            continue

        summary = normalize_string(
            item.get("summary"),
            max_chars=MAX_EVIDENCE_CHARS,
        )

        if not summary:
            continue

        raw_participants = item.get("participants", [])
        participants = []

        if isinstance(raw_participants, list):
            for participant in raw_participants:
                participant = normalize_string(participant)
                if participant and is_valid_mention(participant):
                    participants.append(participant)

        evidence = normalize_string(
            item.get("evidence"),
            max_chars=MAX_EVIDENCE_CHARS,
        )

        events.append(
            {
                "summary": summary,
                "participants": participants,
                "evidence": evidence,
            }
        )

    return events


def default_payload(chunk_text: str) -> dict[str, Any]:
    return {
        "mentions": [],
        "relations": [],
        "themes": [],
        "events": [],
        "evidence_quote": normalize_string(
            chunk_text,
            max_chars=MAX_EVIDENCE_CHARS,
        ),
        "confidence": 0.0,
    }


def normalize_payload(
    data: dict[str, Any],
    chunk_text: str,
) -> dict[str, Any]:
    mentions = []

    raw_mentions = data.get("mentions", [])

    if isinstance(raw_mentions, list):
        for item in raw_mentions:
            mention = normalize_mention(item)
            if mention is not None:
                mentions.append(mention)

    raw_confidence = data.get("confidence", 0.0)

    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    return {
        "mentions": mentions,
        "relations": normalize_relations(data.get("relations", [])),
        "themes": normalize_themes(data.get("themes", [])),
        "events": normalize_events(data.get("events", [])),
        "evidence_quote": normalize_string(
            data.get("evidence_quote"),
            max_chars=MAX_EVIDENCE_CHARS,
        )
        or normalize_string(chunk_text, max_chars=MAX_EVIDENCE_CHARS),
        "confidence": confidence,
    }


def extract_chunk_llm(chunk_text: str) -> dict[str, Any]:
    if not isinstance(chunk_text, str) or not chunk_text.strip():
        return default_payload("")

    prompt = build_prompt(chunk_text)

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": 0,
            },
            format="json",
            think=False,
            keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "15m"),
        )
    except Exception as exc:
        print(f"Ollama error: {exc}")
        return default_payload(chunk_text)

    try:
        content = response["message"]["content"]
    except (KeyError, TypeError):
        return default_payload(chunk_text)

    raw = clean_json(content)

    if not raw:
        return default_payload(chunk_text)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return default_payload(chunk_text)

    if not isinstance(data, dict):
        return default_payload(chunk_text)

    return normalize_payload(data, chunk_text)
