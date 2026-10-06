from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from pathlib import Path

from storage.local import read_json, write_json


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


class AuthService:
    """Local password store and short-lived HS256 bearer tokens."""

    def __init__(self, users_path: str | Path, secret: str, token_ttl_seconds: int = 86400) -> None:
        self.users_path = Path(users_path)
        self.secret = secret.encode("utf-8")
        self.token_ttl_seconds = token_ttl_seconds

    def _users(self) -> dict:
        value = read_json(self.users_path, {})
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _password_hash(password: str, salt: bytes) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)

    def register(self, username: str, password: str) -> None:
        username = username.strip().casefold()
        if not 3 <= len(username) <= 80 or not username.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Nom d’utilisateur invalide.")
        if len(password) < 12 or len(password) > 256:
            raise ValueError("Le mot de passe doit contenir entre 12 et 256 caractères.")
        users = self._users()
        if username in users:
            raise ValueError("Ce nom d’utilisateur existe déjà.")
        salt = secrets.token_bytes(16)
        users[username] = {"salt": _b64(salt), "password_hash": _b64(self._password_hash(password, salt)), "role": "user"}
        write_json(self.users_path, users)

    def login(self, username: str, password: str) -> str:
        username = username.strip().casefold()
        record = self._users().get(username)
        if not record:
            raise ValueError("Identifiants incorrects.")
        salt = base64.urlsafe_b64decode(record["salt"] + "==")
        expected = base64.urlsafe_b64decode(record["password_hash"] + "==")
        if not hmac.compare_digest(self._password_hash(password, salt), expected):
            raise ValueError("Identifiants incorrects.")
        now = int(time.time())
        header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
        body = _b64(json.dumps({"sub": username, "role": record.get("role", "user"), "iat": now, "exp": now + self.token_ttl_seconds}, separators=(",", ":")).encode())
        signing = f"{header}.{body}".encode()
        signature = _b64(hmac.new(self.secret, signing, hashlib.sha256).digest())
        return f"{header}.{body}.{signature}"

    def verify(self, token: str) -> str:
        try:
            header, body, signature = token.split(".")
            signing = f"{header}.{body}".encode()
            expected = _b64(hmac.new(self.secret, signing, hashlib.sha256).digest())
            if not hmac.compare_digest(signature, expected):
                raise ValueError
            payload = json.loads(base64.urlsafe_b64decode(body + "=="))
            if int(payload["exp"]) <= int(time.time()) or payload["sub"] not in self._users():
                raise ValueError
            return str(payload["sub"])
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Session invalide ou expirée.") from exc
