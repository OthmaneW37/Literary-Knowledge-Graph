from __future__ import annotations

import json
import re
from typing import Any

import ollama


MODEL_NAME = "qwen2.5:7b-instruct"

BANNED_MENTION_STARTS = {
    "the",
    "a",
    "an",
    "his",
    "her",
    "their",
    "its",
    "this",
    "that",
    "these",
    "those",
    "he",
    "she",
    "it",
    "they",
    "we",
    "i",
    "you",
}

BANNED_MENTION_CONTAINS = {
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
    "work",
    "time",
    "itch",
    "legs",
    "gas",
    "gaslight",
    "couch",
    "newspaper",
    "alarm",
    "locksmith",
}

VERB_HINTS = {
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
    "leave",
    "keep",
    "speak",
    "tug",
    "throw",
    "sit",
    "sleep",
    "learn",
    "go",
    "come",
    "see",
    "hear",
    "exclaim",
    "ask",
    "say",
}


def build_prompt(chunk_text: str) -> str:
    return f"""
Extract structured literary entities from the passage below.

Return ONLY valid JSON with this schema:
{{
  "mentions": [
    {{
      "text": "surface form exactly as seen in the passage",
      "context": "short local context",
      "chapter": "chapter or section label if available"
    }}
  ],
  "relations": [
    {{
      "source": "exact mention",
      "target": "exact mention",
      "type": "FAMILY|AUTHORITY_OVER|CONFLICT_WITH|ASSISTS|COMMUNICATES_WITH|OBSERVES",
      "evidence": "short verbatim quote"
    }}
  ],
  "themes": [
    {{"name": "alienation"}},
    {{"name": "family"}},
    {{"name": "identity"}},
    {{"name": "authority"}},
    {{"name": "bureaucracy"}},
    {{"name": "guilt"}}
  ],
  "events": [
    {{
      "summary": "short event summary",
      "participants": ["exact mention"],
      "evidence": "short verbatim quote"
    }}
  ],
  "evidence_quote": "short verbatim quote from the passage",
  "confidence": 0.0
}}

Rules:
- Do not resolve coreference manually.
- Do not invent canonical names.
- Extract mentions only as they appear.
- Keep context short.
- Return raw JSON only.

Passage:
{chunk_text}
"""


def clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def default_payload(chunk_text: str) -> dict[str, Any]:
    return {
        "mentions": [],
        "relations": [],
        "themes": [],
        "events": [],
        "evidence_quote": chunk_text[:500],
        "confidence": 0.0,
    }


def is_valid_mention(text: str) -> bool:
    if not isinstance(text, str):
        return False

    t = text.strip()
    if not t:
        return False

    if len(t.split()) > 5:
        return False

    first = t.split()[0].lower().strip(".,;:!?\"'")
    if first in BANNED_MENTION_STARTS:
        return False

    low = t.lower()
    if any(b in low for b in BANNED_MENTION_CONTAINS):
        return False

    if any(v in low.split() for v in VERB_HINTS):
        return False

    if re.search(r"\b(he|she|it|they|his|her|their|them|him)\b", low):
        return False

    if re.search(r"[.,;:!?]{2,}", t):
        return False

    return True


def extract_chunk_llm(chunk_text: str) -> dict[str, Any]:
    prompt = build_prompt(chunk_text)
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
        format="json",
    )

    raw = clean_json(response["message"]["content"])

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return default_payload(chunk_text)

    if not isinstance(data, dict):
        return default_payload(chunk_text)

    data.setdefault("mentions", [])
    data.setdefault("relations", [])
    data.setdefault("themes", [])
    data.setdefault("events", [])
    data.setdefault("evidence_quote", chunk_text[:500])
    data.setdefault("confidence", 0.0)

    mentions = []
    for item in data.get("mentions", []):
        if isinstance(item, dict) and is_valid_mention(item.get("text", "")):
            mentions.append(item)
    data["mentions"] = mentions

    return data