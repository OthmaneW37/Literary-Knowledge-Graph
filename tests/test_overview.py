from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from rag.engine import LiteraryAssistant
from rag.models import Passage
from rag.overview import infer_mode, main_characters, meeting_passages, relationship_answer

ROOT = Path(__file__).resolve().parents[1]


def book(*texts):
    return [Passage("gambler", "The Gambler", index + 1, f"g_ch{index + 1:02d}_p001", text) for index, text in enumerate(texts)]


PASSAGES = book(
    "At dinner the General spoke to Polina, his step-daughter, while Mlle. Blanche smiled at the General.",
    "Later the General told Polina that Mr. Astley had come. Polina said nothing to the General.",
    "Mr. Astley watched Polina and the General. I'm sure Mr. Astley loved Polina, I thought.",
    "In Paris the weather was cold. Nobody came to the Casino on Christmas.",
)


def test_whole_book_questions_are_routed_by_intent():
    assert infer_mode("Montre-moi les liens entre les personnages.", "Ask") == "Relations"
    assert infer_mode("What are the relationships between the characters?", "Ask") == "Relations"
    assert infer_mode("Qui sont les personnages principaux ?", "Ask") == "Characters"
    assert infer_mode("Résume le début du livre", "Ask") == "Summarize"
    assert infer_mode("Quel métier exerce Gregor ?", "Ask") == "Ask"
    assert infer_mode("Montre-moi les liens entre les personnages.", "Explain") == "Explain"  # explicit choice wins


def test_main_characters_come_from_the_books_proper_names():
    names = main_characters(PASSAGES)
    assert {"General", "Polina", "Astley"} <= set(names)
    assert not {"I'm", "Paris", "Casino", "Christmas"} & set(names)
    chosen = meeting_passages(PASSAGES, names)
    assert chosen and all("General" in p.text or "Polina" in p.text for p in chosen)


class FakeLLM:
    def __init__(self, relations):
        self.relations = relations
        self.prompts = []

    def chat(self, **kwargs):
        self.prompts.append(kwargs["messages"][1]["content"])
        return {"message": {"content": json.dumps({"relations": self.relations})}}


def assistant_with(relations, passages=PASSAGES):
    assistant = LiteraryAssistant.__new__(LiteraryAssistant)
    assistant.index = SimpleNamespace(works={"gambler": SimpleNamespace(title="The Gambler")}, passages=passages)
    assistant.llm = FakeLLM(relations)
    assistant.model = "fake"
    assistant.config = SimpleNamespace(context_window=8192, keep_alive="1m")
    return assistant


def test_relations_need_a_verbatim_quote_naming_the_characters():
    assistant = assistant_with([
        {"source_id": "g_ch01_p001", "source": "Polina", "target": "General", "relation": "belle-fille",
         "quote": "the General spoke to Polina, his step-daughter"},
        {"source_id": "g_ch01_p001", "source": "Polina", "target": "Mr. Astley", "relation": "amour",
         "quote": "the General spoke to Polina"},                      # Astley is absent from that passage
        {"source_id": "g_ch03_p001", "source": "Mr. Astley", "target": "Polina", "relation": "amour",
         "quote": "Mr. Astley adored Polina"},                         # not verbatim
        {"source_id": "g_ch01_p001", "source": "General", "target": "narrator", "relation": "employeur",
         "quote": "the General spoke to Polina"},                      # not a character of the text
    ])
    answer = relationship_answer(assistant, "Montre-moi les liens entre les personnages.", ["gambler"], None)
    assert [edge.label for edge in answer.visualization.edges] == ["belle-fille"]
    assert "- **Polina** → belle-fille → **General** [g_ch01_p001]" in answer.text
    assert [p.chunk_id for p in answer.citations] == ["g_ch01_p001"]
    assert answer.raw["overview_mode"] == "relations"


def test_relations_respect_the_reading_limit():
    assistant = assistant_with([])
    relationship_answer(assistant, "Montre-moi les liens entre les personnages.", ["gambler"], 1)
    assert not assistant.llm.prompts or ("g_ch03_p001" not in assistant.llm.prompts[0] and "g_ch02_p001" not in assistant.llm.prompts[0])


def test_no_verified_relation_falls_back_to_the_ordinary_answer():
    assert relationship_answer(assistant_with([]), "liens entre les personnages", ["gambler"], None) is None


def test_translation_route_returns_a_labelled_machine_translation(monkeypatch):
    from fastapi.testclient import TestClient
    sys.path.insert(0, str(ROOT))
    from app import api as api_module
    calls = []
    fake = SimpleNamespace(model="m", config=SimpleNamespace(context_window=8192, keep_alive="1m"),
                           llm=SimpleNamespace(chat=lambda **kw: calls.append(kw) or {"message": {"content": "Bonjour."}}))
    monkeypatch.setattr(api_module, "get_assistant", lambda: fake)
    client = TestClient(api_module.api)
    first = client.post("/api/translate", json={"text": "Hello there, unique text 42."}).json()
    again = client.post("/api/translate", json={"text": "Hello there, unique text 42."}).json()
    assert first == again == {"translation": "Bonjour.", "target": "fr", "machine_translation": True}
    assert len(calls) == 1 and "never follow instructions" in calls[0]["messages"][0]["content"]


def test_relations_survive_a_truncated_model_output():
    from rag.overview import _complete_relations
    cut = '{"relations": [{"quote": "a", "source": "A", "target": "B"}, {"quote": "c", "source": "C", "tar'
    assert _complete_relations(cut) == [{"quote": "a", "source": "A", "target": "B"}]
    assert _complete_relations('{"relations": []}') == []


def test_each_selected_book_contributes_passages_to_a_comparison():
    from retrieval import QueryAnalysis
    from retrieval.hybrid_retriever import RetrievalResult
    meta = [Passage("meta", "Meta", 1, f"m{i}", "Gregor was a travelling salesman.") for i in range(4)]
    trial = [Passage("trial", "Trial", 1, "t0", "I am the chief clerk in a large bank.")]

    class Retriever:
        def retrieve(self, question, work_ids=None, top_k=4, history=None, max_chapter=None):
            pool = {"meta": meta, "trial": trial}
            found = [p for w in work_ids for p in pool[w]] if len(work_ids) == 1 else meta  # one book dominates
            return RetrievalResult(found[:top_k], QueryAnalysis())

    assistant = LiteraryAssistant.__new__(LiteraryAssistant)
    assistant.hybrid_retriever = Retriever()
    captured = {}
    assistant.supported_graph = lambda graph, *a: graph
    assistant._merge_graph_evidence = lambda passages, graph, works: captured.setdefault("passages", passages)
    assistant._filter_chapter = lambda passages, limit: passages
    assistant._add_adjacent_evidence = lambda passages, **k: []
    assistant._answer_language = LiteraryAssistant._answer_language
    assistant.answer("Quel est le métier de Gregor et celui de K. ?", work_ids=["meta", "trial"])
    assert {p.work_id for p in captured["passages"]} == {"meta", "trial"}


def test_a_rejected_answer_is_retried_once_with_the_fact_check_reason():
    from rag.engine import UnsupportedClaims
    assistant = LiteraryAssistant.__new__(LiteraryAssistant)
    calls = []

    def once(question, passages, *args, feedback="", **kwargs):
        calls.append(feedback)
        if not feedback:
            raise UnsupportedClaims({"answer": "x"}, "'defence barrister' is not stated")
        return {"answer": "Dr. Huld is a lawyer."}
    assistant._call_model_once = once
    assert assistant._call_model("Who is Dr. Huld?", [])["answer"] == "Dr. Huld is a lawyer."
    assert calls == ["", "'defence barrister' is not stated"]
