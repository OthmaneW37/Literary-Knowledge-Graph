from rag.engine import LiteraryAssistant
from rag.models import Passage
from retrieval import GraphRetrievalResult, QueryAnalysis, RetrievalResult

from test_local_index import make_index


class StubRetriever:
    def __init__(self, index, graph=None):
        self.lexical = type("Lexical", (), {"index": index})()
        self.graph = graph or GraphRetrievalResult(available=False)

    def retrieve(self, question, work_ids=None, top_k=6, history=None, max_chapter=None):
        return RetrievalResult(
            passages=self.lexical.index.search(
                question, work_ids=work_ids, top_k=top_k, max_chapter=max_chapter
            ),
            analysis=QueryAnalysis(rewritten_query=question),
            graph=self.graph,
        )


def test_answer_rejects_mixture_of_valid_and_invented_citations(monkeypatch, tmp_path) -> None:
    index = make_index(tmp_path)
    assistant = LiteraryAssistant(model="test-model", retriever=StubRetriever(index))
    monkeypatch.setattr(
        assistant,
        "_call_model",
        lambda question, passages, history=None, graph_result=None, analysis=None, mode="Ask", background="": {
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
    assert not answer.used_model
    assert not answer.visualization.is_visible


def test_answer_has_extractive_fallback(monkeypatch, tmp_path) -> None:
    index = make_index(tmp_path)
    assistant = LiteraryAssistant(model="test-model", retriever=StubRetriever(index))
    monkeypatch.setattr(
        assistant,
        "_call_model",
        lambda question, passages, history=None, graph_result=None, analysis=None, mode="Ask": (
            _ for _ in ()
        ).throw(RuntimeError("offline")),
    )

    answer = assistant.answer("Josef arrested", work_ids=["trial"])

    assert not answer.used_model
    assert answer.citations[0].chunk_id == "trial_1"


def test_focused_excerpt_surfaces_relevant_sentence() -> None:
    text = (
        "The room was quiet and the window was open. "
        "Franz and the second warder told Josef that he was under arrest. "
        "Breakfast remained on the table."
    )
    excerpt = LiteraryAssistant._focused_excerpt("Who arrests Josef?", text)

    assert "Franz" in excerpt
    assert "under arrest" in excerpt


def test_family_graph_does_not_invent_a_named_parent_from_cooccurrence() -> None:
    from graph.local_store import LocalGraphStore
    analysis = QueryAnalysis(
        question_type="relationship",
        entities=["Gregor"],
        use_graph=True,
        rewritten_query="Gregor family",
    )
    passage = Passage(
        work_id="meta",
        work_title="Metamorphosis",
        chapter=2,
        chunk_id="meta_family",
        text="Gregor listened while his sister and mother spoke outside his room.",
    )

    record = LocalGraphStore.validate({
        "nodes": [{"name": "Gregor", "kind": "character"}, {"name": "Grete", "kind": "character"}],
        "relations": [{"source": "Grete", "target": "Gregor", "relation": "SISTER_OF", "evidence": passage.text}],
    }, passage)
    assert record["relations"] == []


def test_answer_language_follows_the_question() -> None:
    assert LiteraryAssistant._answer_language("Why is Josef K. arrested?") == "English"
    assert LiteraryAssistant._answer_language("Pourquoi Gregor se cache-t-il ?") == "French"


def test_answer_rejects_generated_answer_without_real_citations(monkeypatch, tmp_path) -> None:
    index = make_index(tmp_path)
    assistant = LiteraryAssistant(model="test-model", retriever=StubRetriever(index))
    monkeypatch.setattr(
        assistant,
        "_call_model",
        lambda question, passages, history=None, graph_result=None, analysis=None, mode="Ask", background="": {
            "answer": "A fact with no valid citation.", "citation_ids": ["made_up_chunk"]
        },
    )

    answer = assistant.answer("Josef arrested", work_ids=["trial"])

    assert not answer.used_model
    assert "source" in answer.raw["error"].casefold()


def test_answer_retrieval_respects_spoiler_chapter(tmp_path) -> None:
    index = make_index(tmp_path)
    assistant = LiteraryAssistant(model="test-model", retriever=StubRetriever(index))

    passages = assistant.retrieve("Josef", work_ids=["trial"], max_chapter=1)

    assert all(int(passage.chapter) <= 1 for passage in passages)


def test_retrieval_gives_clear_out_of_scope_message(tmp_path) -> None:
    index = make_index(tmp_path)
    assistant = LiteraryAssistant(model="test-model", retriever=StubRetriever(index))

    answer = assistant.answer("quantum spaceship in the book", work_ids=["trial"])

    assert not answer.used_model
    assert "enough evidence" in answer.text.casefold()
    assert not answer.citations
