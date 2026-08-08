from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Work:
    work_id: str
    title: str
    author: str
    language: str = ""


@dataclass(frozen=True)
class Passage:
    work_id: str
    work_title: str
    chapter: int | str
    chunk_id: str
    text: str
    score: float = 0.0

    @property
    def citation_label(self) -> str:
        return f"{self.work_title}, chapitre {self.chapter} — {self.chunk_id}"


@dataclass(frozen=True)
class GraphNode:
    id: str
    label: str
    kind: str = "character"


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    label: str
    evidence_chunk_id: str


@dataclass
class Visualization:
    type: str = "none"
    title: str = ""
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    @property
    def is_visible(self) -> bool:
        return self.type != "none" and bool(self.nodes)


@dataclass
class Answer:
    text: str
    citations: list[Passage]
    visualization: Visualization = field(default_factory=Visualization)
    used_model: bool = True
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
