from __future__ import annotations

from typing import Any, TypedDict


class NarrativeState(TypedDict, total=False):
    user_id: str
    question: str
    work_ids: list[str]
    max_chapter: int | None
    image_analysis: dict[str, Any] | None
    answer: Any
    status: str
    clarification: str | None
    uncertainty: list[str]
    top_k: int
    history: list[dict[str, str]]
    mode: str
    retrieval_query: str
