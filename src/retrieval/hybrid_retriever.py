# src/retrieval/hybrid_retriever.py

from __future__ import annotations

from dataclasses import dataclass, field

from rag.models import Passage

from .graph_retriever import (
    GraphRetriever,
    GraphRetrievalResult,
)

from .lexical_retriever import LexicalRetriever

from .query_analyzer import (
    QueryAnalyzer,
    QueryAnalysis,
)


@dataclass
class RetrievalResult:

    passages: list[Passage]

    analysis: QueryAnalysis

    graph: GraphRetrievalResult = field(
        default_factory=GraphRetrievalResult
    )


class HybridRetriever:
    """
    Orchestrateur de retrieval.

    Pipeline :

    question
        ↓
    QueryAnalyzer
        ↓
    ┌───────────────┐
    │               │
    BM25          Neo4j
    │               │
    └──────┬────────┘
           ↓
    RetrievalResult
    """

    def __init__(
        self,
        lexical: LexicalRetriever,
        analyzer: QueryAnalyzer,
        graph: GraphRetriever | None = None,
    ) -> None:

        self.lexical = lexical
        self.analyzer = analyzer
        self.graph = graph

    def retrieve(
        self,
        question: str,
        work_ids: list[str] | None = None,
        top_k: int = 6,
        history: list[dict[str, str]] | None = None,
    ) -> RetrievalResult:

        # -----------------------------------
        # 1. Comprendre la question
        # -----------------------------------

        analysis = self.analyzer.analyze(
            question,
            history=history,
        )

        # -----------------------------------
        # 2. Retrieval lexical
        # -----------------------------------

        passages: list[Passage] = []

        if analysis.use_lexical:

            passages = self.lexical.retrieve(
                analysis.rewritten_query,
                work_ids=work_ids,
                top_k=top_k,
            )

        # -----------------------------------
        # 3. Retrieval graphe
        # -----------------------------------

        graph_result = GraphRetrievalResult(
            available=False
        )

        if (
            analysis.use_graph
            and self.graph
            and analysis.entities
        ):

            graph_result = self.graph.retrieve(
                entities=analysis.entities,
                work_ids=work_ids,
            )

        # -----------------------------------
        # 4. Retour structuré
        # -----------------------------------

        return RetrievalResult(
            passages=passages,
            analysis=analysis,
            graph=graph_result,
        )