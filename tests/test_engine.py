from rag.engine import LiteraryAssistant
from retrieval import GraphRetrievalResult, QueryAnalysis, RetrievalResult

from test_local_index import make_index


class StubRetriever:
    def __init__(self, index, graph=None):
        self.lexical = type("Lexical", (), {"index": index})()
        self.graph = graph or GraphRetrievalResult(available=False)

    def retrieve(self, question, work_ids=None, top_k=6, history=None):
        return RetrievalResult(
            passages=self.lexical.index.search(question, work_ids=work_ids, top_k=top_k),
            analysis=QueryAnalysis(rewritten_query=question),
            graph=self.graph,
        )


def test_answer_validates_citations_and_graph(monkeypatch, tmp_path) -> None:
    index = make_index(tmp_path)
    assistant = LiteraryAssistant(model="test-model", retriever=StubRetriever(index))
    monkeypatch.setattr(
        assistant,
        "_call_model",
        lambda question, passages, history=None, graph_result=None, analysis=None: {
            "answer": "Josef K. est arrêté. [trial_1] [invented]",
            "citation_ids": ["trial_1", "invented"],
            "visualization": {
                "type": "relationship_graph",
                "title": "Arrestation",
                "nodes": [
                    {"id": "k", "label": "Josef K.", "kind": "character"},
                    {"id": "warders", "label": "Warders", "kind": "character"},
                ],
                "edges": [
                    {"source": "warders", "target": "k", "label": "arrêtent", "evidence_chunk_id": "trial_1"},
                    {"source": "k", "target": "warders", "label": "invalid", "evidence_chunk_id": "invented"},
                ],
            },
        },
    )

    answer = assistant.answer("Who arrested Josef?", work_ids=["trial"])

    assert [passage.chunk_id for passage in answer.citations] == ["trial_1"]
    assert len(answer.visualization.edges) == 1


def test_answer_has_extractive_fallback(monkeypatch, tmp_path) -> None:
    index = make_index(tmp_path)
    assistant = LiteraryAssistant(model="test-model", retriever=StubRetriever(index))
    monkeypatch.setattr(
        assistant,
        "_call_model",
        lambda question, passages, history=None, graph_result=None, analysis=None: (
            _ for _ in ()
        ).throw(RuntimeError("offline")),
    )

    answer = assistant.answer("Josef arrested", work_ids=["trial"])

    assert not answer.used_model
    assert answer.citations[0].chunk_id == "trial_1"
