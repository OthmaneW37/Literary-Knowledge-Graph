"""Regression coverage for incomplete graphs, source isolation and resumable builds."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import api as api_module
from graph.local_store import LocalGraphStore
from rag.engine import LiteraryAssistant
from rag.models import Passage
from retrieval.graph_retriever import GraphRetrievalResult, GraphRelationship
from storage.local import write_json


def assistant_for(tmp_path, passages):
    assistant = LiteraryAssistant.__new__(LiteraryAssistant)
    assistant.index = SimpleNamespace(
        passages=passages,
        works={p.work_id: SimpleNamespace(title=p.work_title, author="Writer", language="en") for p in passages},
        get_passage=lambda key: next((p for p in passages if p.chunk_id == key), None),
        stats=lambda: {},
    )
    store = LocalGraphStore(assistant.index, tmp_path / "graphs")
    assistant.hybrid_retriever = SimpleNamespace(graph=SimpleNamespace(local_store=store))
    assistant.model = "test"
    assistant.config = SimpleNamespace(embedding_model="test")
    return assistant, store


def save_record(store, passage, names, relations=None):
    data = store._read(passage.work_id)
    data[passage.chunk_id] = store.validate({"nodes": [{"name": n, "kind": "character"} for n in names],
                                           "relations": relations or []}, passage)
    write_json(store._path(passage.work_id), data)


def test_isolated_characters_have_sources_and_respect_chapter_limit(tmp_path):
    first = Passage("book", "Book", 1, "one", "Alice walked into the quiet room.")
    later = Passage("book", "Book", 2, "two", "Bernard arrived the following day.")
    assistant, store = assistant_for(tmp_path, [first, later])
    save_record(store, first, ["Alice"])
    save_record(store, later, ["Bernard"])
    graph = api_module.graph_dict(store.explore(["book"]), assistant, ["book"], 1)
    assert [n["label"] for n in graph["nodes"]] == ["Alice"]
    assert graph["nodes"][0]["mentions"][0]["chunk_id"] == "one"
    assert graph["edges"] == []
    assert graph["coverage"] == {"processed": 1, "total": 1, "remaining": 0}


def test_homonyms_in_different_books_are_not_merged(tmp_path):
    passages = [Passage(w, w, 1, w, "Alice spoke to Bernard.") for w in ("book1", "book2")]
    assistant, store = assistant_for(tmp_path, passages)
    for passage in passages:
        save_record(store, passage, ["Alice", "Bernard"], [{"source": "Alice", "target": "Bernard",
                     "relation": "parle à", "evidence": passage.text}])
    graph = api_module.graph_dict(store.explore(), assistant, ["book1", "book2"], None)
    assert len(graph["nodes"]) == 4
    assert len({n["id"] for n in graph["nodes"]}) == 4
    assert graph["edges"][0]["source"] != graph["edges"][1]["source"]


def test_unambiguous_short_name_is_consolidated_but_ambiguous_one_is_not(tmp_path):
    p = Passage("book", "Book", 1, "one", "Gregor Samsa spoke. Gregor waited. Alice Smith met Alice Jones. Alice waved.")
    assistant, store = assistant_for(tmp_path, [p])
    save_record(store, p, ["Gregor Samsa", "Gregor", "Alice", "Alice Smith", "Alice Jones"])
    graph = api_module.graph_dict(store.explore(), assistant, ["book"], None)
    labels = [n["label"] for n in graph["nodes"]]
    assert "Gregor" not in labels and "Gregor Samsa" in labels
    assert "Alice" in labels and "Alice Smith" in labels and "Alice Jones" in labels


def test_graph_does_not_silently_truncate_after_150_relations(tmp_path):
    passages = [Passage("book", "Book", 2 if i < 151 else 1, str(i), "Alice spoke to Bernard.") for i in range(152)]
    assistant, store = assistant_for(tmp_path, passages)
    for p in passages:
        save_record(store, p, ["Alice", "Bernard"], [{"source": "Alice", "target": "Bernard", "relation": "parle à", "evidence": p.text}])
    assert len(store.explore().relationships) == 152
    limited = store.explore(["book"], limit=1, max_chapter=1)
    assert limited.relationships[0].evidence_chunk_id == "151"


def test_build_resumes_without_reprocessing_and_reports_coverage(tmp_path):
    passages = [Passage("book", "Book", i + 1, str(i), "Alice went outside.") for i in range(3)]
    _, store = assistant_for(tmp_path, passages)
    calls = []
    def chat(**kwargs):
        calls.append(kwargs)
        return {"message": {"content": json.dumps({"nodes": [{"name": "Alice", "kind": "character"}], "relations": []})}}
    provider = SimpleNamespace(chat=chat)
    first = store.build(provider, "test", ["book"], limit=1, max_chapter=2)
    assert first == {"processed": 1, "remaining": 1, "total": 2}
    second = store.build(provider, "test", ["book"], limit=10, max_chapter=2)
    assert second["processed"] == 1 and second["remaining"] == 0
    assert len(calls) == 2
    assert store.coverage(["book"]) == {"processed": 2, "total": 3, "remaining": 1}
    passages[0] = replace(passages[0], text=passages[0].text + " A new revision.")
    assert store.coverage(["book"])["remaining"] == 2
    assert all(n["evidence_chunk_id"] != "0" for n in store.explore().nodes)


def test_failed_build_keeps_successful_passages_and_can_resume(tmp_path):
    passages = [Passage("book", "Book", 1, str(i), "Alice went outside.") for i in range(2)]
    _, store = assistant_for(tmp_path, passages)
    responses = iter(['{"nodes": [], "relations": []}', '{"nodes":'])
    provider = SimpleNamespace(chat=lambda **_: {"message": {"content": next(responses)}})
    with pytest.raises(ValueError):
        store.build(provider, "test", ["book"], limit=2)
    assert store.coverage(["book"])["processed"] == 1
    assert store.coverage(["book"])["remaining"] == 1


def test_names_are_complete_mentions_not_substrings():
    p = Passage("book", "Book", 1, "one", "Anna stood with her mother at the station.")
    result = LocalGraphStore.validate({"nodes": [{"name": "Ann", "kind": "character"},
                                                {"name": "mother", "kind": "character"}], "relations": []}, p)
    assert result["nodes"] == {"mother": "character"}


@pytest.mark.parametrize("payload", [{}, {"nodes": None, "relations": []}, {"nodes": [], "relations": {}}])
def test_malformed_extraction_is_not_marked_complete(payload):
    with pytest.raises(ValueError):
        LocalGraphStore.validate(payload, Passage("b", "B", 1, "p", "Alice waited."))


def test_unknown_api_path_is_json_404():
    response = TestClient(api_module.api).get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_relations_cannot_borrow_quotes_from_other_characters_or_books(tmp_path):
    p = Passage("book", "Book", 1, "one", "Alice spoke to Bernard.")
    assistant, _ = assistant_for(tmp_path, [p])
    graph = GraphRetrievalResult(relationships=[
        GraphRelationship("Alice", "Clara", "speaks to", "one", p.text, "book"),
        GraphRelationship("Alice", "Bernard", "speaks to", "one", p.text, "another_book"),
    ])
    assert not assistant.supported_graph(graph, ["book"]).relationships


def test_health_distinguishes_api_from_unavailable_ollama(tmp_path, monkeypatch):
    import httpx
    assistant, _ = assistant_for(tmp_path, [])
    assistant.config.ollama_host = "http://127.0.0.1:11434"
    assistant.hybrid_retriever.graph.available = False
    monkeypatch.setattr(api_module, "get_assistant", lambda: assistant)
    def offline(*args, **kwargs):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(api_module.httpx, "get", offline)
    response = TestClient(api_module.api).get("/api/health")
    assert response.json()["ok"] is True
    assert response.json()["ollama"] is False


def test_every_inline_citation_requires_its_own_quote(tmp_path):
    passages = [Passage("book", "Book", 1, "one", "Alice is a teacher."),
                Passage("book", "Book", 1, "two", "Bernard is a doctor.")]
    assistant, _ = assistant_for(tmp_path, passages)
    with pytest.raises(ValueError, match="own exact"):
        assistant._validate_answer({"answer": "Alice teaches and practices medicine [one] [two]",
                                    "citation_ids": ["one", "two"], "evidence_quotes": {"one": passages[0].text},
                                    "_require_quotes": True}, passages)


def test_grounding_audit_rejects_a_role_attributed_to_the_wrong_person(tmp_path):
    passage = Passage("book", "Book", 1, "one", "Alice is a teacher. Bernard is a doctor.")
    assistant, _ = assistant_for(tmp_path, [passage])
    assistant.config.context_window = 8192
    assistant.config.keep_alive = "15m"
    requests = []
    def chat(**kwargs):
        requests.append(kwargs)
        return {"message": {"content": '{"supported": false, "reason": "Doctor refers to Bernard, not Alice."}'}}
    assistant.llm = SimpleNamespace(chat=chat)
    with pytest.raises(ValueError, match="correct character"):
        assistant._verify_grounding("What is Alice's job?", {"answer": "Alice is a doctor [one]", "citation_ids": ["one"]}, [passage])
    assert "Alice is a teacher" in requests[0]["messages"][1]["content"]


def test_relation_review_removes_rejected_claims_but_keeps_character_mentions(tmp_path):
    passage = Passage("book", "Book", 1, "one", "Alice feeds Bernard every morning.")
    _, store = assistant_for(tmp_path, [passage])
    save_record(store, passage, ["Alice", "Bernard"], [
        {"source": "Alice", "target": "Bernard", "relation": "feeds", "evidence": passage.text},
        {"source": "Bernard", "target": "Alice", "relation": "feeds", "evidence": passage.text},
    ])
    provider = SimpleNamespace(chat=lambda **_: {"message": {"content": '{"accepted": [0, 900, -1, true]}'}})
    assert store.audit(provider, "test", ["book"])["rejected_relations"] == 1
    result = store.explore()
    assert len(result.nodes) == 2
    assert len(result.relationships) == 1 and result.relationships[0].source == "Alice"
    assert store.audit(provider, "test", ["book"])["reviewed_passages"] == 0


def test_character_discovery_recovers_omissions_without_inventing_names(tmp_path):
    p = Passage("book", "Book", 1, "one", 'Alice called, "Anna! Anna!" The maid answered.')
    later = Passage("book", "Book", 2, "two", "Mr. Bernard arrived.")
    assistant, store = assistant_for(tmp_path, [p, later])
    save_record(store, p, ["Alice"])
    save_record(store, later, [])
    calls = []
    def chat(**kwargs):
        calls.append(kwargs)
        return {"message": {"content": '{"names": ["Anna", "Bernard", "Ghost"]}'}}
    provider = SimpleNamespace(chat=chat)
    assert store.discover_characters(provider, "test", ["book"]) == 2
    visible = api_module.graph_dict(store.explore(), assistant, ["book"], 1)
    assert {n["label"] for n in visible["nodes"]} == {"Alice", "Anna"}
    assert store.discover_characters(provider, "test", ["book"]) == 0
    assert len(calls) == 1


@pytest.mark.parametrize("method,path,payload", [
    ("get", "/api/graph?work_ids=book", None),
    ("get", "/api/books/book/passages", None),
    ("post", "/api/graph/build", {"work_ids": ["book"]}),
    ("post", "/api/library/archive/book", None),
    ("post", "/api/search", {"query": "Alice", "work_ids": ["book"]}),
])
def test_protected_book_routes_require_login(monkeypatch, method, path, payload):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    response = TestClient(api_module.api).request(method, path, json=payload)
    assert response.status_code == 401


def test_archived_books_remain_visible_for_restoration(tmp_path, monkeypatch):
    assistant, _ = assistant_for(tmp_path, [])
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setattr(api_module, "get_assistant", lambda: assistant)
    monkeypatch.setattr(api_module.library, "archived_ids", lambda: {"archived"})
    monkeypatch.setattr(api_module.library, "records", lambda: [{"work_id": "archived", "title": "An archived book"}])
    response = TestClient(api_module.api).get("/api/library")
    assert response.status_code == 200
    assert response.json()["works"][0]["archived"] is True
