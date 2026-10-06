from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

from storage.local import read_json, write_json


class AnnotationService:
    """Approval-bound, user-scoped annotation storage with idempotent saves."""

    def __init__(self, data_dir: str | Path, secret: str | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.secret = (secret or os.getenv("APPROVAL_SECRET") or os.getenv("JWT_SECRET") or secrets.token_urlsafe(32)).encode("utf-8")

    def _user_dir(self, user_id: str) -> Path:
        key = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]
        return self.data_dir / "library/users" / key

    @staticmethod
    def _payload(user_id: str, work_id: str, chapter: int, text: str, evidence_ids: list[str]) -> dict:
        if not user_id or not work_id or chapter < 1 or not text.strip() or len(text) > 4000:
            raise ValueError("Annotation invalide.")
        evidence_ids = list(dict.fromkeys(evidence_ids))
        if not evidence_ids or len(evidence_ids) > 20:
            raise ValueError("Au moins une preuve vérifiable est requise.")
        return {"user_id": user_id, "work_id": work_id, "chapter": chapter, "text": text.strip(), "evidence_ids": evidence_ids}

    def prepare(self, user_id: str, work_id: str, chapter: int, text: str, evidence_ids: list[str]) -> dict:
        payload = self._payload(user_id, work_id, chapter, text, evidence_ids)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        expires = int(time.time()) + 600
        nonce = secrets.token_urlsafe(12)
        message = f"{digest}:{expires}:{nonce}".encode()
        token = f"{digest}.{expires}.{nonce}.{hmac.new(self.secret, message, hashlib.sha256).hexdigest()}"
        return {"action": "save_annotation", "work_id": work_id, "chapter": chapter,
                "text": text.strip(), "evidence_ids": payload["evidence_ids"], "approval_token": token}

    def save(self, user_id: str, work_id: str, chapter: int, text: str, evidence_ids: list[str], approval_token: str) -> dict:
        payload = self._payload(user_id, work_id, chapter, text, evidence_ids)
        try:
            digest, expires_s, nonce, signature = approval_token.split(".")
            expires = int(expires_s)
        except (ValueError, AttributeError) as exc:
            raise ValueError("Approbation invalide. Préparez à nouveau l’annotation.") from exc
        expected_digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        message = f"{digest}:{expires}:{nonce}".encode()
        expected_signature = hmac.new(self.secret, message, hashlib.sha256).hexdigest()
        if expires <= int(time.time()) or not hmac.compare_digest(digest, expected_digest) or not hmac.compare_digest(signature, expected_signature):
            raise ValueError("Approbation expirée ou contenu modifié. Préparez à nouveau l’annotation.")
        record_id = hashlib.sha256(f"{user_id}:{digest}".encode()).hexdigest()[:32]
        path = self._user_dir(user_id) / "annotations.json"
        records = read_json(path, [])
        if not isinstance(records, list):
            records = []
        existing = next((item for item in records if item.get("id") == record_id), None)
        if existing:
            return {"saved": True, "id": record_id, "already_saved": True}
        records.append({"id": record_id, **payload, "created_at": int(time.time())})
        write_json(path, records)
        return {"saved": True, "id": record_id, "already_saved": False}

    def list(self, user_id: str, work_id: str | None = None) -> list[dict]:
        records = read_json(self._user_dir(user_id) / "annotations.json", [])
        return [item for item in records if item.get("work_id") == work_id] if work_id else records
