from types import SimpleNamespace
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import api as api_module
from rag.models import Answer, Passage, Visualization


class FakeIndex:
    def __init__(self):
        self.works = {"demo": SimpleNamespace(title="A Demo Novel", author="A. Writer", language="en")}
        self.passages = [Passage(
            work_id="demo", work_title="A Demo Novel", chapter=1,
            chunk_id="demo_ch01_001", text="A sufficiently long passage for a small API test.",
            author="A. Writer", language="en",
        )]

    def stats(self):
        return {"demo": len(self.passages)}

    def get_passage(self, chunk_id):
        return next((p for p in self.passages if p.chunk_id == chunk_id), None)


class FakeAssistant:
    def __init__(self):
        self.index = FakeIndex()
        self.model = "local-test-model"
        self.config = SimpleNamespace(embedding_model="embedding-test")
        self.hybrid_retriever = SimpleNamespace(semantic=None)

    def answer(self, question, **kwargs):
        return Answer(
            text=f"Evidence for: {question}", citations=self.index.passages,
            visualization=Visualization(), retrieval_ms=3, generation_ms=5,
        )


def test_library_and_passage_routes_serialize_local_index(monkeypatch):
    assistant = FakeAssistant()
    monkeypatch.setattr(api_module, "get_assistant", lambda: assistant)
    client = TestClient(api_module.api)

    library_response = client.get("/api/library")
    assert library_response.status_code == 200
    assert library_response.json()["works"][0]["id"] == "demo"

    passage_response = client.get("/api/books/demo/passages?chapter=1")
    assert passage_response.status_code == 200
    assert passage_response.json()["passages"][0]["citation_label"].startswith("A Demo Novel")


def test_chat_route_uses_local_answer_and_persists_conversation(monkeypatch):
    assistant = FakeAssistant()
    monkeypatch.setattr(api_module, "get_assistant", lambda: assistant)
    monkeypatch.setattr(api_module.conversations, "save", lambda scope, messages, conversation_id=None: "a" * 32)
    client = TestClient(api_module.api)

    response = client.post("/api/chat", json={"question": "What is this?", "work_ids": ["demo"]})

    assert response.status_code == 200
    assert response.json()["content"] == "Evidence for: What is this?"
    assert response.json()["citations"][0]["chunk_id"] == "demo_ch01_001"
    assert response.json()["conversation_id"] == "a" * 32
