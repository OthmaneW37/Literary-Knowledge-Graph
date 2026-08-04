from __future__ import annotations

import json
import re
from typing import Any

import ollama

MODEL_NAME = "qwen2.5:7b-instruct"


def build_prompt(chunk_text: str) -> str:
    return f"""
You are extracting literary information from a novel passage.

Return ONLY valid JSON with this schema:
{{
  "characters": ["..."],
  "relations": [
    {{
      "source": "...",
      "target": "...",
      "type": "FAMILY|AUTHORITY_OVER|CONFLICT_WITH|ASSISTS|COMMUNICATES_WITH|OBSERVES",
      "evidence": "..."
    }}
  ],
  "themes": ["alienation", "bureaucracy", "family", "guilt", "identity", "authority"],
  "events": [
    {{
      "summary": "...",
      "participants": ["..."],
      "evidence": "..."
    }}
  ],
  "evidence_quote": "...",
  "confidence": 0.0
}}

Rules:
- Use only names explicitly present or strongly implied in the text.
- Keep evidence short and verbatim.
- If nothing is found, return empty lists.
- Confidence must be between 0 and 1.
- Output raw JSON only. No markdown. No code fences.

Passage:
{chunk_text}
"""


def _clean_json_text(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"```$", "", text.strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _default_payload(chunk_text: str) -> dict[str, Any]:
    return {
        "characters": [],
        "relations": [],
        "themes": [],
        "events": [],
        "evidence_quote": chunk_text[:500],
        "confidence": 0.0,
    }


def call_llm(prompt: str, chunk_text: str) -> dict[str, Any]:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
        format="json",
    )

    content = response["message"]["content"]
    content = _clean_json_text(content)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return _default_payload(chunk_text)

    if not isinstance(data, dict):
        return _default_payload(chunk_text)

    data.setdefault("characters", [])
    data.setdefault("relations", [])
    data.setdefault("themes", [])
    data.setdefault("events", [])
    data.setdefault("evidence_quote", chunk_text[:500])
    data.setdefault("confidence", 0.0)

    return data


def extract_chunk_llm(chunk_text: str) -> dict[str, Any]:
    prompt = build_prompt(chunk_text)
    return call_llm(prompt, chunk_text)