from __future__ import annotations

import subprocess
import sys

from retrieval import (
    GraphRetriever,
    GraphRelationship,
    GraphRetrievalResult,
    HybridRetriever,
    QueryAnalysis,
    QueryAnalyzer,
)
from retrieval.query_analyzer import expand_bilingual_query
from extraction.entity_resolution import EntityCandidate
from graph.load_neo4j import extraction_matches_chunk, match_existing_entity


def test_retrieval_package_imports_without_cycle() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "from retrieval import HybridRetriever, QueryAnalyzer"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_relationship_type_always_enables_graph() -> None:
    analysis = QueryAnalysis.from_dict(
        {
            "question_type": "relationship",
            "entities": ["Gregor"],
            "use_lexical": "true",
            "use_graph": "false",
        },
        original_question="Relation de Gregor avec son père",
    )
    assert analysis.use_lexical is True
    assert analysis.use_graph is True


def test_heuristic_analyzer_extracts_named_character() -> None:
    analysis = QueryAnalyzer()._heuristic_analysis(
        "Montre la relation entre Gregor Samsa et son père."
    )
    assert analysis.question_type == "relationship"
    assert analysis.use_graph is True
    assert "Gregor Samsa" in analysis.entities


def test_heuristic_analyzer_removes_english_possessive_from_character_name():
    analysis = QueryAnalyzer()._heuristic_analysis("What is Josef K.’s profession?")
    assert analysis.entities == ["Josef K"]


def test_default_analyzer_is_fast_and_expands_french_without_llm(monkeypatch) -> None:
    monkeypatch.setattr(
        "retrieval.query_analyzer.ollama.chat",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )
    analysis = QueryAnalyzer().analyze(
        "Pourquoi Gregor cache sa transformation à sa famille ?"
    )

    assert "why" in analysis.rewritten_query
    assert "family" in analysis.rewritten_query
    assert "Gregor" in analysis.rewritten_query
    assert analysis.entities == ["Gregor"]


def test_bilingual_expansion_also_supports_english_question_on_french_text() -> None:
    expanded = expand_bilingual_query("Why does the father feel guilt?")
    assert "pourquoi" in expanded
    assert "pere" in expanded
    assert "culpabilite" in expanded


def test_hybrid_retriever_calls_graph_for_relationship() -> None:
    class Analyzer:
        def analyze(self, question, history=None):
            return QueryAnalysis(
                question_type="relationship",
                entities=["Gregor"],
                use_lexical=True,
                use_graph=True,
                rewritten_query="Gregor father relationship",
            )

    class Lexical:
        def retrieve(self, query, work_ids=None, top_k=6, max_chapter=None):
            assert query == "Gregor father relationship"
            return []

    class Graph:
        def retrieve(self, entities, work_ids=None):
            assert entities == ["Gregor"]
            return GraphRetrievalResult(
                characters=["Gregor", "Father"],
                relationships=[
                    GraphRelationship("Father", "Gregor", "PARENT_OF", "meta_1")
                ],
                evidence_chunk_ids=["meta_1"],
            )

    result = HybridRetriever(Lexical(), Analyzer(), Graph()).retrieve(
        "Qui est le père de Gregor?",
        work_ids=["meta"],
    )
    assert result.graph.relationships[0].relation == "PARENT_OF"


def test_hybrid_retriever_fuses_lexical_and_semantic_rankings() -> None:
    from rag.models import Passage

    passages = {
        chunk_id: Passage("work", "Work", 1, chunk_id, chunk_id)
        for chunk_id in ("lexical", "shared", "semantic")
    }

    class Analyzer:
        def analyze(self, question, history=None):
            return QueryAnalysis(rewritten_query=question)

    class Lexical:
        def retrieve(self, query, work_ids=None, top_k=6, max_chapter=None):
            return [passages["lexical"], passages["shared"]]

    class Semantic:
        def retrieve(self, query, work_ids=None, top_k=6, max_chapter=None):
            return [passages["semantic"], passages["shared"]]

    result = HybridRetriever(
        Lexical(), Analyzer(), semantic=Semantic(), min_semantic_score=0
    ).retrieve("question", top_k=3)

    assert result.passages[0].chunk_id == "shared"
    assert {passage.chunk_id for passage in result.passages} == set(passages)


def test_hybrid_retriever_rejects_low_signal_unrelated_question() -> None:
    from rag.models import Passage

    class Analyzer:
        def analyze(self, question, history=None):
            return QueryAnalysis(rewritten_query=question)

    class Lexical:
        def retrieve(self, query, work_ids=None, top_k=6, max_chapter=None):
            return [Passage("work", "Book", 1, "irrelevant", "A long literary scene about an argument.", 5.0)]

    class Semantic:
        def retrieve(self, query, work_ids=None, top_k=6, max_chapter=None):
            return [Passage("work", "Book", 1, "irrelevant", "An argument.", 0.29)]

    result = HybridRetriever(
        Lexical(), Analyzer(), semantic=Semantic(),
        min_semantic_score=0.40, min_lexical_coverage=0.30,
    ).retrieve("purple spaceship argument novel", top_k=3)

    assert result.passages == []


def test_hybrid_retriever_rejects_named_character_absent_from_selected_book() -> None:
    from types import SimpleNamespace
    from rag.models import Passage, Work

    passage = Passage("kafka", "The Trial", 1, "kafka_ch01_p001", "Josef K. was arrested one morning.")
    index = SimpleNamespace(
        works={"kafka": Work("kafka", "The Trial", "Franz Kafka", "en")},
        passages=[passage],
    )

    class Analyzer:
        def analyze(self, question, history=None):
            return QueryAnalysis(entities=["Raskolnikov"], rewritten_query=question)

    class Lexical:
        def __init__(self):
            self.index = index

        def retrieve(self, query, work_ids=None, top_k=6, max_chapter=None):
            return []

    class Semantic:
        def retrieve(self, query, work_ids=None, top_k=6, max_chapter=None):
            return [Passage("kafka", "The Trial", 1, "kafka_ch01_p001", passage.text, 0.48)]

    result = HybridRetriever(
        Lexical(), Analyzer(), semantic=Semantic(), min_semantic_score=0.4
    ).retrieve("Why does Raskolnikov commit the murder?", work_ids=["kafka"])

    assert result.passages == []


def test_graph_retriever_reads_canonical_schema() -> None:
    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, cypher, **parameters):
            assert "source.canonical_name" in cypher
            assert "entity CONTAINS" in cypher
            assert parameters["entities"] == ["gregor"]
            return [
                {
                    "source": "Gregor's father",
                    "target": "Gregor Samsa",
                    "relation": "FAMILY",
                    "evidence_chunk_id": "meta_14",
                }
            ]

    class Driver:
        def session(self):
            return Session()

    retriever = GraphRetriever.__new__(GraphRetriever)
    retriever.driver = Driver()
    result = retriever.retrieve(["Gregor"], work_ids=["meta"])
    assert result.available is True
    assert result.relationships[0].target == "Gregor Samsa"
    assert result.evidence_chunk_ids == ["meta_14"]


def test_graph_retriever_stays_off_unless_enabled() -> None:
    retriever = GraphRetriever(password="configured-but-disabled", enabled=False)
    result = retriever.retrieve(["Gregor"])

    assert result.available is False


def test_relation_endpoint_matches_existing_character_only() -> None:
    entities = [
        EntityCandidate(
            canonical_name="Gregor Samsa",
            aliases={"Gregor"},
        )
    ]
    assert match_existing_entity("Gregor", entities) == "Gregor Samsa"
    assert match_existing_entity("his room", entities) is None


def test_stale_extraction_is_rejected() -> None:
    extraction = {"evidence_quote": "Gregor woke from troubled dreams"}
    assert extraction_matches_chunk(
        extraction,
        "One morning Gregor woke from troubled dreams and found himself transformed.",
    )
    assert not extraction_matches_chunk(
        extraction,
        "Josef K. was arrested one morning without having done anything wrong.",
    )
