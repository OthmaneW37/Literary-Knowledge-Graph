from __future__ import annotations

from typing import Any

import ollama
from security.budget import consume


class OllamaProvider:
    """Small provider boundary so application logic is not tied to a global Ollama client."""

    name = "ollama"

    def __init__(self, host: str | None = None, timeout_seconds: float = 120.0) -> None:
        self.host = host
        self.client = ollama.Client(
            host=host or None,
            timeout=max(1.0, timeout_seconds),
        )

    def chat(self, **kwargs: Any) -> Any:
        budget = consume('model')
        if budget:
            client = ollama.Client(host=self.host or None, timeout=budget.remaining())
            try:
                return client.chat(**kwargs)
            finally:
                client._client.close()
        return self.client.chat(**kwargs)

    def embed(self, **kwargs: Any) -> Any:
        budget = consume('model')
        if budget:
            client = ollama.Client(host=self.host or None, timeout=budget.remaining())
            try:
                return client.embed(**kwargs)
            finally:
                client._client.close()
        return self.client.embed(**kwargs)


def create_provider(
    name: str,
    host: str | None = None,
    timeout_seconds: float = 120.0,
) -> OllamaProvider:
    normalized = name.strip().casefold()
    if normalized == "ollama":
        return OllamaProvider(host=host, timeout_seconds=timeout_seconds)
    raise ValueError(f"LLM provider non pris en charge : {name}")
