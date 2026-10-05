from __future__ import annotations

import os
from dataclasses import dataclass


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(value, maximum))


@dataclass(frozen=True)
class RAGConfig:
    """Runtime settings with conservative limits for a responsive local chat."""

    model: str
    provider: str
    ollama_host: str
    ollama_timeout_seconds: float
    temperature: float
    embedding_model: str
    embedding_batch_size: int
    min_semantic_score: float
    min_lexical_coverage: float
    query_analysis: str
    context_window: int
    max_output_tokens: int
    max_context_chars: int
    keep_alive: str
    rerank: bool = True

    @classmethod
    def from_env(cls, model: str | None = None) -> "RAGConfig":
        analysis = os.getenv("RAG_QUERY_ANALYSIS", "heuristic").strip().casefold()
        if analysis not in {"heuristic", "llm"}:
            analysis = "heuristic"
        return cls(
            model=model or os.getenv("OLLAMA_MODEL", "qwen3.5:4b-q4_K_M"),
            provider=os.getenv("LLM_PROVIDER", "ollama").strip().casefold() or "ollama",
            ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip(),
            ollama_timeout_seconds=_bounded_float("OLLAMA_TIMEOUT_SECONDS", 120.0, 1.0, 600.0),
            temperature=_bounded_float("RAG_TEMPERATURE", 0.0, 0.0, 2.0),
            embedding_model=os.getenv(
                "OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b"
            ).strip(),
            embedding_batch_size=_bounded_int("RAG_EMBED_BATCH_SIZE", 16, 1, 128),
            min_semantic_score=_bounded_float("RAG_MIN_SEMANTIC_SCORE", 0.40, 0.0, 1.0),
            min_lexical_coverage=_bounded_float("RAG_MIN_LEXICAL_COVERAGE", 0.30, 0.0, 1.0),
            query_analysis=analysis,
            context_window=_bounded_int("RAG_NUM_CTX", 8192, 2048, 32768),
            max_output_tokens=_bounded_int("RAG_NUM_PREDICT", 700, 128, 2000),
            max_context_chars=_bounded_int("RAG_MAX_CONTEXT_CHARS", 18000, 4000, 60000),
            keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "15m").strip() or "15m",
            rerank=os.getenv("RAG_RERANK", "true").casefold() in {"1", "true", "yes"},
        )


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(value, maximum))
