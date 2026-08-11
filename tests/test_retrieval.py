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
        def retrieve(self, query, work_ids=None, top_k=6):
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


def test_graph_retriever_reads_canonical_schema() -> None:
    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, cypher, **parameters):
            assert "source.canonical_name" in cypher
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
