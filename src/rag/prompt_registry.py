from __future__ import annotations

import json
from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(name: str, fallback: str) -> tuple[str, str]:
    """Read a JSON-compatible YAML prompt record without adding a YAML dependency."""
    path = PROMPT_DIR / f"{name}.yaml"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        return str(record["prompt"]), str(record.get("version", name))
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return fallback, f"{name}-fallback"
