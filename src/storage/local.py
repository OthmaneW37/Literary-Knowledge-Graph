from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Fichier local illisible : {path.name}. Restaurez-le depuis une sauvegarde.") from exc


def write_json(path: Path, value) -> None:
    """Publish a complete JSON document using an atomic same-directory replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ConversationStore:
    def __init__(self, directory: str | Path = "data/library/conversations") -> None:
        self.directory = Path(directory)

    def _path(self, conversation_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", conversation_id):
            raise ValueError("Identifiant de discussion invalide.")
        return self.directory / f"{conversation_id}.json"

    def save(self, scope: str, messages: list[dict], conversation_id: str | None = None) -> str:
        conversation_id = conversation_id or uuid4().hex
        title = next((m["content"][:80] for m in messages if m.get("role") == "user"), "Nouvelle discussion")
        write_json(self._path(conversation_id), {
            "id": conversation_id, "scope": scope, "title": title,
            "updated_at": datetime.now(timezone.utc).isoformat(), "messages": messages,
        })
        return conversation_id

    def list(self, scope: str) -> list[dict]:
        results = []
        for path in self.directory.glob("*.json"):
            try:
                record = read_json(path, {})
                if record.get("scope") == scope:
                    results.append(record)
            except ValueError:
                continue
        return sorted(results, key=lambda item: item.get("updated_at", ""), reverse=True)

    def load(self, conversation_id: str, scope: str) -> list[dict]:
        record = read_json(self._path(conversation_id), {})
        if record.get("scope") != scope:
            raise ValueError("Cette discussion appartient à un autre livre ou niveau de lecture.")
        return record.get("messages", [])

    def archive(self, conversation_id: str) -> None:
        source = self._path(conversation_id)
        destination = self.directory / "archive" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            source.replace(destination)
