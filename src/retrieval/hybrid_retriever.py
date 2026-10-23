# src/retrieval/hybrid_retriever.py

from __future__ import annotations

import difflib
import logging
import os
import re
from dataclasses import dataclass, field, replace

from rag.models import Passage
from rag.local_index import tokenize, normalize_text

from .graph_retriever import (
    GraphRetriever,
    GraphRetrievalResult,
)

from .lexical_retriever import LexicalRetriever
from .semantic_retriever import SemanticRetriever
from .reranker import diversify
from .spoiler_policy import SpoilerPolicy

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
    ┌──────────┬─────────────┐
    │          │             │
    BM25   embeddings      Neo4j
    │          │             │
    └──── fusion RRF ────────┘
               ↓
    RetrievalResult
    """

    def __init__(
        self,
        lexical: LexicalRetriever,
        analyzer: QueryAnalyzer,
        graph: GraphRetriever | None = None,
        semantic: SemanticRetriever | None = None,
        min_semantic_score: float = 0.40,
        min_lexical_coverage: float = 0.30,
        rerank: bool = False,
        learned_reranker=None,
        expander=None,
        llm_reranker=None,
    ) -> None:

        self.expander = expander
        self.llm_reranker = llm_reranker
        self.lexical = lexical
        self.analyzer = analyzer
        self.graph = graph
        self.semantic = semantic
        self.min_semantic_score = min_semantic_score
        self.min_lexical_coverage = min_lexical_coverage
        self.rerank = rerank
        self.learned_reranker = learned_reranker

    def retrieve(
        self,
        question: str,
        work_ids: list[str] | None = None,
        top_k: int = 6,
        history: list[dict[str, str]] | None = None,
        max_chapter: int | None = None,
    ) -> RetrievalResult:
        index = getattr(self.lexical, "index", None)
        if index:
            eligible = set(index.works if work_ids is None else work_ids)
            named_works = [work_id for work_id, work in index.works.items()
                           if work_id in eligible and len(work.title) > 3
                           and normalize_text(work.title) in normalize_text(question)]
            if named_works:
                work_ids = named_works

        # -----------------------------------
        # 1. Comprendre la question
        # -----------------------------------

        analysis = self.analyzer.analyze(
            question,
            history=history,
        )
        respelled = self._respell_names(analysis.entities, work_ids)
        if respelled:
            # "Aliocha" in a French question is "Alyosha" in the English text.
            analysis.rewritten_query = f"{analysis.rewritten_query} {' '.join(respelled.values())}"
            analysis.entities = [" ".join(respelled.get(t, t) for t in tokenize(e)) for e in analysis.entities]

        # -----------------------------------
        # 2. Retrieval lexical
        # -----------------------------------

        depth = top_k * 2
        lexical_passages: list[Passage] = []

        if analysis.use_lexical:

            lexical_passages = self.lexical.retrieve(
                analysis.rewritten_query,
                work_ids=work_ids,
                top_k=depth,
                max_chapter=max_chapter,
            )

        # Search probes in the edition's language (keywords + hypothetical
        # sentence). They widen recall only; evidence still comes from the text.
        expansion = None
        if self.expander is not None and index is not None:
            languages = {index.works[w].language for w in (work_ids or index.works) if w in index.works}
            expansion = self.expander.expand(question, languages.pop() if len(languages) == 1 else "en")
        expanded_lexical: list[Passage] = []
        if expansion and expansion.keywords and analysis.use_lexical:
            expanded_lexical = self.lexical.retrieve(
                expansion.keywords, work_ids=work_ids, top_k=depth, max_chapter=max_chapter,
            )

        semantic_passages: list[Passage] = []
        extra_semantic: list[list[Passage]] = []
        if self.semantic:
            semantic_query = question + (" " + " ".join(analysis.entities) if history else "")
            try:
                if hasattr(self.semantic, "retrieve_many"):
                    semantic_passages, *extra_semantic = self.semantic.retrieve_many(
                        [semantic_query], work_ids=work_ids, top_k=depth, max_chapter=max_chapter,
                        documents=[expansion.passage] if expansion and expansion.passage else [],
                    )
                else:
                    semantic_passages = self.semantic.retrieve(
                        semantic_query, work_ids=work_ids, top_k=depth, max_chapter=max_chapter,
                    )
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Semantic retrieval unavailable; using BM25 only: %s", exc
                )

        if not self._has_sufficient_evidence(
            analysis.rewritten_query,
            lexical_passages,
            semantic_passages,
            entities=analysis.entities,
            work_ids=work_ids,
            expanded=(expansion.keywords, expanded_lexical) if expansion else None,
        ):
            passages = []
        else:
            if self.llm_reranker is not None:
                pool = self._fuse_rankings(
                    [lexical_passages, semantic_passages, expanded_lexical, *extra_semantic],
                    max(self.llm_reranker.candidates, top_k),
                )
                try:
                    passages = self.llm_reranker.rerank(question, pool, top_k, probe=expansion.keywords if expansion else "")
                except Exception as exc:
                    logging.getLogger(__name__).warning("LLM reranker failed; using local diversification: %s", exc)
                    passages = diversify(pool, top_k)
            else:
                passages = self._fuse_rankings(
                    [lexical_passages, semantic_passages, expanded_lexical, *extra_semantic],
                    top_k * 2 if self.rerank else top_k,
                )
            # The targeted rules below (profession, introductions, openings)
            # apply after either reranker.
            if self.rerank and self.llm_reranker is None:
                reranker = self.learned_reranker
                provider = os.getenv("RERANKER_PROVIDER", "diversify").casefold()
                if reranker is None and provider in {"cross_encoder", "trained"}:
                    try:
                        if provider == "trained":
                            from reranking.trained import TrainedReranker
                            reranker = TrainedReranker(os.getenv("RERANKER_PATH", "models/reranker"))
                        else:
                            from reranking.cross_encoder import CrossEncoderReranker
                            reranker = CrossEncoderReranker(os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"))
                        self.learned_reranker = reranker
                    except Exception as exc:
                        logging.getLogger(__name__).warning("Learned reranker unavailable; using local diversification: %s", exc)
                if reranker is not None:
                    try:
                        passages = reranker.rerank(question, passages, top_k)
                    except Exception as exc:
                        logging.getLogger(__name__).warning("Reranker failed; using local diversification: %s", exc)
                        passages = diversify(passages, top_k)
                else:
                    passages = diversify(passages, top_k)
            if index and re.search(r"\b(profession|occupation|job|métier|metier|employment)\b", question, re.I):
                # Job descriptions are frequently in quoted introductions ("I am a...")
                # rather than sentences repeating a full character name.
                relevant_works = {p.work_id for p in passages[:3]}
                entity_aliases = [entity for entity in analysis.entities]
                alias_terms = {term for alias in entity_aliases for term in tokenize(alias)}
                alias_patterns = [
                    re.compile(rf"\b{re.escape(term)}\b", re.I)
                    for term in alias_terms if len(term) >= 3
                ]
                # Initials such as Josef K. are discarded by ordinary word
                # tokenization; retain an explicit capitalized initial to find
                # first-person employment statements in the surrounding text.
                alias_patterns.extend(
                    re.compile(rf"(?<!\w){re.escape(initial)}\.")
                    for alias in entity_aliases
                    for initial in re.findall(r"(?<!\w)([A-Z])\.?", alias)
                )
                declarations = [
                    re.compile(rf"\b(?:{re.escape(alias)})\b.{{0,60}}\b(?:is|was|works? as|worked as|est|était|travaille comme)\s+(?:a|an|the|un|une|le|la)\s+", re.I)
                    for alias in entity_aliases if len(alias) >= 3
                ]
                # Also match short declarative formulations such as "Samsa was a
                # travelling salesman" when retrieval extracted "Gregor Samsa".
                surname_declarations = [
                    re.compile(rf"\b{re.escape(term)}\b[^.!?]{{0,60}}\b(?:is|was|est|était)\s+(?:a|an|the|un|une|le|la)\s+[^.!?]{{2,45}}", re.I)
                    for term in alias_terms if len(term) >= 3
                ]
                candidates = [p for p in index.passages if p.work_id in relevant_works
                              and SpoilerPolicy.allows(p.chapter, max_chapter)
                              and (any(pattern.search(p.text) for pattern in declarations + surname_declarations)
                                   or (any(pattern.search(p.text) for pattern in alias_patterns)
                                       and re.search(r"\b(?:I am|you are|he is|she is|je suis)\s+(?:a|an|the|un|une|le|la)\s+", p.text, re.I)))]
                names = set(tokenize(question))
                direct_self_introduction = re.compile(
                    r"\b(?:I am|je suis)\s+(?:a|an|the|un|une|le|la)\s+", re.I
                )
                candidates.sort(key=lambda p: (
                    not any(pattern.search(p.text) for pattern in declarations + surname_declarations),
                    not bool(direct_self_introduction.search(p.text)),
                    int(p.chapter) if str(p.chapter).isdigit() else 999999,
                    p.chunk_index,
                    -len(alias_terms & set(tokenize(p.text))), -len(names & set(tokenize(p.text))),
                ))
                added = candidates[:2]
                added_ids = {p.chunk_id for p in added}
                passages = (added + [p for p in passages if p.chunk_id not in added_ids])[:top_k]
            if index and re.search(r"\b(who is|who was|describe|qui est|qui était|décris|decris)\b", question, re.I):
                # For character introductions, later plot mentions are often
                # lexically stronger than the first passage that identifies them.
                relevant_works = list(dict.fromkeys(p.work_id for p in passages))[:2]
                aliases = [term for entity in analysis.entities for term in tokenize(entity)]
                introductions = [
                    p for p in index.passages
                    if p.work_id in relevant_works
                    and SpoilerPolicy.allows(p.chapter, max_chapter)
                    and any(re.search(rf"\b{re.escape(term)}\b", p.text, re.I) for term in aliases if len(term) >= 2)
                ]
                introductions.sort(key=lambda p: (p.work_id, p.chapter if isinstance(p.chapter, int) else 999999, p.chunk_index))
                if introductions:
                    first = introductions[0]
                    passages = [first] + [p for p in passages if p.chunk_id != first.chunk_id][:top_k - 1]
            if index and re.search(r"\b(beginning|opening|début|commencement)\b", question, re.I):
                relevant_works = list(dict.fromkeys(p.work_id for p in passages))[:2]
                openings = []
                for work_id in relevant_works:
                    opening = next((p for p in index.passages if p.work_id == work_id and p.chapter == 1), None)
                    if opening and SpoilerPolicy.allows(opening.chapter, max_chapter):
                        openings.append(opening)
                opening_ids = {p.chunk_id for p in openings}
                passages = (openings + [p for p in passages if p.chunk_id not in opening_ids])[:top_k]

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
            if max_chapter is not None:
                graph_result.relationships = [
                    relation
                    for relation in graph_result.relationships
                    if (
                        self._evidence_chapter(relation.evidence_chunk_id) is not None
                        and self._evidence_chapter(relation.evidence_chunk_id) <= max_chapter
                    )
                ]
                graph_result.evidence_chunk_ids = [
                    relation.evidence_chunk_id
                    for relation in graph_result.relationships
                    if relation.evidence_chunk_id
                ]

        # -----------------------------------
        # 4. Retour structuré
        # -----------------------------------

        return RetrievalResult(
            passages=passages,
            analysis=analysis,
            graph=graph_result,
        )

    @staticmethod
    def _reciprocal_rank_fusion(
        lexical: list[Passage],
        semantic: list[Passage],
        top_k: int,
        rank_constant: int = 60,
    ) -> list[Passage]:
        """Fuse ranks without assuming BM25 and cosine scores share a scale."""
        return HybridRetriever._fuse_rankings([lexical, semantic], top_k, rank_constant)

    @staticmethod
    def _fuse_rankings(rankings: list[list[Passage]], top_k: int, rank_constant: int = 60) -> list[Passage]:
        rankings = [ranking for ranking in rankings if ranking]
        if len(rankings) <= 1:
            return (rankings[0] if rankings else [])[:top_k]
        scores: dict[str, float] = {}
        passages: dict[str, Passage] = {}
        first_position: dict[str, int] = {}
        for result_list in rankings:
            for rank, passage in enumerate(result_list, start=1):
                passages[passage.chunk_id] = passage
                first_position.setdefault(passage.chunk_id, len(first_position))
                scores[passage.chunk_id] = scores.get(passage.chunk_id, 0.0) + 1 / (
                    rank_constant + rank
                )
        ranked_ids = sorted(
            scores,
            key=lambda chunk_id: (-scores[chunk_id], first_position[chunk_id]),
        )
        return [
            replace(passages[chunk_id], score=round(scores[chunk_id], 6))
            for chunk_id in ranked_ids[:top_k]
        ]

    def _has_sufficient_evidence(
        self,
        query: str,
        lexical: list[Passage],
        semantic: list[Passage],
        entities: list[str] | None = None,
        work_ids: list[str] | None = None,
        expanded: tuple[str, list[Passage]] | None = None,
    ) -> bool:
        index = getattr(self.lexical, "index", None)
        # A bare initial such as "K" has no searchable term; it cannot prove
        # that a character is absent from the selected books.
        entities = [entity for entity in entities or [] if tokenize(entity)]
        if entities and index is not None:
            selected = set(index.works if work_ids is None else work_ids)
            selected_passages = [
                passage for passage in index.passages if passage.work_id in selected
            ]
            selected_titles = [
                work.title for work_id, work in index.works.items() if work_id in selected
            ]
            entity_present = False
            for entity in entities:
                terms = set(tokenize(entity))
                if terms and (
                    any(terms <= set(tokenize(title)) for title in selected_titles)
                    or any(terms <= set(tokenize(passage.text)) for passage in selected_passages)
                ):
                    entity_present = True
                    break
            if not entity_present and expanded:
                # A translated spelling ("Aliocha" for "Alyosha") is absent as
                # typed; accept it when every proper name of the edition-language
                # rewrite exists in the books and the rewrite matches a passage.
                names = {term for word in re.findall(r"\b[A-Z][\w'’-]+", expanded[0]) for term in tokenize(word)}
                corpus_terms = set().union(*(set(tokenize(p.text)) for p in expanded[1])) if expanded[1] else set()
                entity_present = bool(names) and names <= corpus_terms and self._covers(expanded[0], expanded[1])
            # Vector similarity alone can be deceptively high for a name
            # absent from all selected novels (e.g. Raskolnikov in Kafka).
            if not entity_present and (not semantic or semantic[0].score < 0.65):
                return False

        query_terms = set(tokenize(query))
        if not query_terms:
            return bool(lexical or (semantic and semantic[0].score >= self.min_semantic_score))
        for passage in lexical:
            coverage = len(query_terms & set(tokenize(passage.text))) / len(query_terms)
            if coverage >= self.min_lexical_coverage:
                return True
        if expanded and self._covers(expanded[0], expanded[1]):
            return True
        return bool(semantic and semantic[0].score >= self.min_semantic_score)

    def _respell_names(self, entities: list[str], work_ids: list[str] | None) -> dict[str, str]:
        """Map name spellings absent from the selected books to their closest spelling in them.

        Only names (not ordinary words) are respelled, only within the same
        first letter and similar length, so an absent character (Raskolnikov in
        Kafka) finds no neighbour and is still refused.
        """
        index = getattr(self.lexical, "index", None)
        postings = getattr(index, "_postings", None)
        if not entities or not postings:
            return {}
        selected = set(index.works if work_ids is None else work_ids)
        passages = index.passages

        def in_books(term: str) -> int:
            return sum(1 for position in postings.get(term, ()) if passages[position].work_id in selected)

        respelled = {}
        for term in {t for entity in entities for t in tokenize(entity) if len(t) >= 4}:
            if in_books(term):
                continue
            candidates = [
                (difflib.SequenceMatcher(None, term, word).ratio(), in_books(word), word)
                for word in postings if word[:1] == term[:1] and abs(len(word) - len(term)) <= 2
            ]
            best = max((c for c in candidates if c[0] >= 0.7 and c[1]), default=None)
            if best:
                respelled[term] = best[2]
        return respelled

    def _covers(self, keywords: str, passages: list[Passage]) -> bool:
        """Generated keywords are broader than the question, so they must match more strictly."""
        terms = set(tokenize(keywords))
        return bool(terms) and any(
            len(terms & set(tokenize(passage.text))) / len(terms) >= max(0.5, self.min_lexical_coverage)
            for passage in passages
        )

    @staticmethod
    def _evidence_chapter(chunk_id: str | None) -> int | None:
        import re

        match = re.search(r"_ch(\d+)_", chunk_id or "", re.IGNORECASE)
        return int(match.group(1)) if match else None
