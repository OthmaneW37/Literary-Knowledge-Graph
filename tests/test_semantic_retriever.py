from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rag.models import Passage
from retrieval import SemanticRetriever


def make_index():
    passages = [
        Passage("book_a", "Book A", 1, "a_1", "A man is troubled by guilt."),
        Passage("book_b", "Book B", 2, "b_1", "A family welcomes a joyful return."),
    ]
    return SimpleNamespace(passages=passages, works={"book_a": object(), "book_b": object()})


def test_semantic_cache_embeds_only_missing_passages(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def embed(**kwargs):
        texts = kwargs["input"]
        calls.append(texts)
        return {"embeddings": [[1.0, float(index + 1)] for index, _ in enumerate(texts)]}

    monkeypatch.setattr("retrieval.semantic_retriever.ollama.embed", embed)
    first = SemanticRetriever(make_index(), cache_dir=tmp_path, batch_size=8)
    assert first.prepare() == 2
    assert len(calls) == 1

    calls.clear()
    second = SemanticRetriever(make_index(), cache_dir=tmp_path, batch_size=8)
    assert second.prepare() == 2
    assert calls == []


def test_semantic_retrieval_filters_selected_work(monkeypatch, tmp_path) -> None:
    def embed(**kwargs):
        texts = kwargs["input"]
        if len(texts) == 1 and texts[0].startswith("Instruct:"):
            return {"embeddings": [[1.0, 0.0]]}
        return {"embeddings": [[1.0, 0.0], [0.0, 1.0]]}

    monkeypatch.setattr("retrieval.semantic_retriever.ollama.embed", embed)
    retriever = SemanticRetriever(make_index(), cache_dir=tmp_path, batch_size=8)
    results = retriever.retrieve("guilt", work_ids=["book_a"], top_k=2)

    assert [passage.chunk_id for passage in results] == ["a_1"]
    assert np.isclose(results[0].score, 1.0)
