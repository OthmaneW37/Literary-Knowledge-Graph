from __future__ import annotations

import json
import re
from time import perf_counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .models import Answer, GraphEdge, GraphNode, Passage, Visualization
from .config import RAGConfig
from .local_index import tokenize
from .prompts import QA_SYSTEM, MODE_PROMPTS
from ingestion.loaders import detect_language
from graph.local_store import LocalGraphStore
from llm import LLMProvider, create_provider

from retrieval import (
    GraphRetrievalResult,
    GraphRetriever,
    HybridRetriever,
    LexicalRetriever,
    QueryAnalysis,
    QueryAnalyzer,
    SemanticRetriever,
)

load_dotenv()

def _clean_json(value: str) -> dict[str, Any]:
    value = value.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?", "", value, flags=re.IGNORECASE).strip()
        value = re.sub(r"```$", "", value).strip()
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("The model did not return a JSON object")
    return json.loads(re.sub(r",\s*([}\]])", r"\1", value[start : end + 1]))


class LiteraryAssistant:
    def __init__(
        self,
        manifest_path: str | Path = "data/annotations/work_manifest.json",
        processed_dir: str | Path = "data/processed",
        model: str | None = None,
        retriever: HybridRetriever | None = None,
        config: RAGConfig | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self.config = config or RAGConfig.from_env(model)
        self.model = self.config.model
        self.llm = llm_provider or create_provider(
            self.config.provider,
            host=self.config.ollama_host,
            timeout_seconds=self.config.ollama_timeout_seconds,
        )
        if retriever is None:
            lexical = LexicalRetriever(manifest_path, processed_dir)
            semantic = None
            if self.config.embedding_model:
                semantic = SemanticRetriever(
                    lexical.index,
                    model=self.config.embedding_model,
                    cache_dir=Path(manifest_path).parent.parent / "library" / "embeddings",
                    batch_size=self.config.embedding_batch_size,
                    keep_alive=self.config.keep_alive,
                    embedding_provider=self.llm,
                )
            retriever = HybridRetriever(
                lexical=lexical,
                analyzer=QueryAnalyzer(
                    self.model,
                    mode=self.config.query_analysis,
                    keep_alive=self.config.keep_alive,
                    provider=self.llm,
                ),
                graph=GraphRetriever(local_store=LocalGraphStore(lexical.index, Path(manifest_path).parent.parent / "library/graphs")),
                semantic=semantic,
                min_semantic_score=self.config.min_semantic_score,
                min_lexical_coverage=self.config.min_lexical_coverage,
                rerank=self.config.rerank,
            )
        self.hybrid_retriever = retriever
        # Kept as a public compatibility alias for the Streamlit sidebar and
        # callers that inspect works or corpus statistics.
        self.index = retriever.lexical.index

    def retrieve(
        self,
        question: str,
        work_ids: list[str] | None = None,
        top_k: int = 4,
        history: list[dict[str, str]] | None = None,
        max_chapter: int | None = None,
    ) -> list[Passage]:
        result = self.hybrid_retriever.retrieve(
            question,
            work_ids=work_ids,
            top_k=top_k,
            history=history,
            max_chapter=max_chapter,
        )
        result.graph = self.supported_graph(result.graph, work_ids, max_chapter)
        passages = self._merge_graph_evidence(result.passages, result.graph, work_ids)
        return self._filter_chapter(passages, max_chapter)

    def answer(
        self,
        question: str,
        work_ids: list[str] | None = None,
        top_k: int = 4,
        history: list[dict[str, str]] | None = None,
        max_chapter: int | None = None,
        mode: str = "Ask",
    ) -> Answer:
        if max_chapter is not None:
            history = [item for item in (history or [])[-12:] if item.get("role") == "user"]
        retrieval_started = perf_counter()
        retrieval = self.hybrid_retriever.retrieve(
            question,
            work_ids=work_ids,
            top_k=top_k,
            history=history,
            max_chapter=max_chapter,
        )
        retrieval.graph = self.supported_graph(retrieval.graph, work_ids, max_chapter)
        passages = self._merge_graph_evidence(retrieval.passages, retrieval.graph, work_ids)
        passages = self._filter_chapter(passages, max_chapter)
        if mode.casefold() in {"summarize", "compare", "characters"}:
            passages = self._coverage_passages(question, work_ids, max_chapter, passages, top_k + 2)
        passages = self._add_adjacent_evidence(
            passages,
            max_passages=top_k + 2,
            max_chapter=max_chapter,
        )
        retrieval_ms = round((perf_counter() - retrieval_started) * 1000)
        if not passages:
            return Answer(
                text=(
                    "Je ne trouve pas suffisamment d’éléments dans le texte pour répondre avec certitude."
                    if self._answer_language(question) == "French"
                    else "I couldn’t find enough evidence in the selected text to answer reliably."
                ),
                citations=[],
                used_model=False,
                retrieval_ms=retrieval_ms,
            )

        generation_started = perf_counter()
        try:
            payload = self._call_model(
                question,
                passages,
                history,
                graph_result=retrieval.graph,
                analysis=retrieval.analysis,
                mode=mode,
            )
            answer = self._validate_answer(payload, passages)
            answer.retrieval_ms = retrieval_ms
            answer.generation_ms = round((perf_counter() - generation_started) * 1000)
            if not answer.visualization.is_visible and retrieval.analysis.use_graph:
                graph_visualization = self._visualization_from_graph(retrieval.graph, passages)
                if graph_visualization.is_visible:
                    answer.visualization = graph_visualization
                    cited = {passage.chunk_id for passage in answer.citations}
                    passages_by_id = {passage.chunk_id: passage for passage in passages}
                    for edge in graph_visualization.edges:
                        if edge.evidence_chunk_id not in cited:
                            answer.citations.append(passages_by_id[edge.evidence_chunk_id])
                            cited.add(edge.evidence_chunk_id)
            return answer
        except Exception as exc:
            answer = self._extractive_fallback(question, passages, exc)
            answer.retrieval_ms = retrieval_ms
            answer.generation_ms = round((perf_counter() - generation_started) * 1000)
            return answer

    def supported_graph(self, graph, work_ids=None, max_chapter=None):
        """Reject stale, cross-book and unproven relations before any display."""
        selected = set(self.index.works if work_ids is None else work_ids)
        relations = []
        for relation in graph.relationships:
            passage = self.index.get_passage(relation.evidence_chunk_id or "")
            if not passage or passage.work_id not in selected:
                continue
            if max_chapter is not None and (not str(passage.chapter).isdigit() or int(passage.chapter) > max_chapter):
                continue
            quote = re.sub(r"\s+", " ", relation.evidence).strip()
            if not quote or quote not in re.sub(r"\s+", " ", passage.text):
                continue
            relations.append(relation)
        return GraphRetrievalResult(
            characters=sorted({name for r in relations for name, kind in [(r.source, r.source_kind), (r.target, r.target_kind)] if kind == "character"}),
            relationships=relations, evidence_chunk_ids=list(dict.fromkeys(r.evidence_chunk_id for r in relations)), available=graph.available,
        )

    def _add_adjacent_evidence(
        self,
        passages: list[Passage],
        max_passages: int,
        max_chapter: int | None = None,
    ) -> list[Passage]:
        """Repair chunk-boundary gaps around the two strongest results."""
        # Preserve every retrieved/graph-backed passage before adding context;
        # a neighbor must never displace a Neo4j evidence chunk.
        expanded = list(passages[:max_passages])
        known_ids = {passage.chunk_id for passage in expanded}
        original_ids = {passage.chunk_id for passage in passages}
        for passage in passages[:2]:
            if len(expanded) >= max_passages:
                break
            # Prefer the following passage: answers are often stated just
            # after a scene or chapter boundary.
            for neighbor in reversed(self.index.get_neighbors(passage.chunk_id)):
                allowed_chapter = (
                    max_chapter is None
                    or (str(neighbor.chapter).isdigit() and int(neighbor.chapter) <= max_chapter)
                )
                if (
                    allowed_chapter
                    and neighbor.chunk_id not in known_ids
                    and neighbor.chunk_id not in original_ids
                ):
                    expanded.append(neighbor)
                    known_ids.add(neighbor.chunk_id)
                    break
        return expanded

    @staticmethod
    def _filter_chapter(
        passages: list[Passage],
        max_chapter: int | None,
    ) -> list[Passage]:
        if max_chapter is None:
            return passages
        return [
            passage for passage in passages
            if str(passage.chapter).isdigit() and int(passage.chapter) <= max_chapter
        ]

    def _merge_graph_evidence(
        self,
        passages: list[Passage],
        graph_result: GraphRetrievalResult,
        work_ids: list[str] | None,
    ) -> list[Passage]:
        merged = list(passages)
        known_ids = {passage.chunk_id for passage in merged}
        selected_works = set(self.index.works if work_ids is None else work_ids)
        for chunk_id in graph_result.evidence_chunk_ids:
            passage = self.index.get_passage(chunk_id)
            if passage and passage.work_id in selected_works and chunk_id not in known_ids:
                merged.append(passage)
                known_ids.add(chunk_id)
        return merged

    def _call_model(
        self,
        question: str,
        passages: list[Passage],
        history: list[dict[str, str]] | None = None,
        graph_result: GraphRetrievalResult | None = None,
        analysis: QueryAnalysis | None = None,
        mode: str = "Ask",
    ) -> dict[str, Any]:
        language = self._answer_language(question)
        included = []
        context = []
        length = 0
        for passage in passages:
            source = f"[SOURCE {passage.chunk_id}]\nTitle: {passage.work_title}\nChapter: {passage.chapter}\nPage: {passage.page or 'n/a'}\n{passage.text}"
            if length + len(source) > self.config.max_context_chars:
                continue
            context.append(source)
            included.append(passage.chunk_id)
            length += len(source)
        if not included:
            raise ValueError("No complete source fits in the configured context budget")
        prompt = (
            f"MODE: {mode}\n{MODE_PROMPTS.get(mode.casefold(), MODE_PROMPTS['ask'])}\n"
            f"QUESTION: {question}\n"
            f"RECENT CONVERSATION (not evidence): {json.dumps((history or [])[-6:], ensure_ascii=False)}\n"
            f"ALLOWED SOURCE IDS: {', '.join(included)}\n\n" + "\n\n".join(context)
        )
        response = self.llm.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": QA_SYSTEM.format(language=language)},
                {"role": "user", "content": prompt},
            ],
            format={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "citation_ids": {"type": "array", "items": {"type": "string", "enum": included}, "maxItems": 3},
                    "evidence_quotes": {"type": "object", "properties": {item: {"type": "string"} for item in included}, "additionalProperties": False},
                    "insufficient_evidence": {"type": "boolean"},
                },
                "required": ["answer", "citation_ids", "evidence_quotes", "insufficient_evidence"],
                "additionalProperties": False,
            }, think=False,
            options={"temperature": self.config.temperature, "num_ctx": self.config.context_window,
                     "num_predict": self.config.max_output_tokens},
            keep_alive=self.config.keep_alive,
        )
        payload = _clean_json(response["message"]["content"])
        # These fields are set by our code, never trusted from model output.
        payload["_context_ids"] = included
        payload["_require_quotes"] = True
        payload["_language"] = language
        return payload

    def _coverage_passages(self, question, work_ids, max_chapter, ranked, limit):
        """Sample across the requested chapters/works for overview modes."""
        selected = set(self.index.works if work_ids is None else work_ids)
        chapter_match = re.search(r"(?:chapters?|chapitres?)\s+(\d+)(?:\s*(?:to|through|à|-)\s*(\d+))?", question, re.I)
        lower = int(chapter_match[1]) if chapter_match else None
        upper = int(chapter_match[2] or chapter_match[1]) if chapter_match else None
        candidates = [
            p for p in self.index.passages if p.work_id in selected
            and str(p.chapter).isdigit()
            and (max_chapter is None or int(p.chapter) <= max_chapter)
            and (lower is None or lower <= int(p.chapter) <= upper)
        ]
        if not candidates:
            return []
        # Round robin ensures that a longer book cannot consume every slot.
        groups = [[p for p in candidates if p.work_id == work_id] for work_id in sorted(selected)]
        chosen = []
        per_work = max(1, limit // max(1, len(groups)))
        for group in groups:
            if not group:
                continue
            positions = sorted({round(i * (len(group) - 1) / max(1, per_work - 1)) for i in range(min(per_work, len(group)))})
            chosen.extend(group[i] for i in positions)
        return chosen[:limit]

    @staticmethod
    def _answer_language(question: str) -> str:
        words = set(re.findall(r"[^\W_]+", question.casefold()))
        if words & {"why", "who", "what", "find", "explain", "summarize", "describe", "tell", "how"}:
            return "English"
        if words & {"pourquoi", "qui", "quoi", "trouve", "explique", "résume", "décris", "comment"}:
            return "French"
        return "French" if detect_language(question) == "fr" else "English"

    @staticmethod
    def _focused_excerpt(question: str, text: str, max_chars: int = 900) -> str:
        """Select query-overlapping sentences so key facts are not buried."""
        query_terms = set(tokenize(question))
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()
        ]
        ranked: list[tuple[int, int, str]] = []
        for position, sentence in enumerate(sentences):
            overlap = len(query_terms & set(tokenize(sentence)))
            if overlap:
                ranked.append((overlap, position, sentence))
        selected = sorted(ranked, key=lambda item: (-item[0], item[1]))[:3]
        if not selected:
            return text[:max_chars]
        selected_positions = {
            nearby
            for _, position, _ in selected
            for nearby in (max(0, position - 1), position)
        }
        excerpt = " ".join(
            sentence for position, sentence in enumerate(sentences) if position in selected_positions
        )
        return excerpt[:max_chars]

    def _validate_answer(self, payload: dict[str, Any], passages: list[Passage]) -> Answer:
        context_ids = set(payload.get("_context_ids", [p.chunk_id for p in passages]))
        allowed = {passage.chunk_id: passage for passage in passages if passage.chunk_id in context_ids}
        answer_text = str(payload.get("answer", "")).strip()
        if not answer_text:
            raise ValueError("The model returned an empty answer")

        if payload.get("insufficient_evidence") is True:
            language = payload.get("_language", "French")
            return Answer(
                text="Je ne trouve pas suffisamment d’éléments dans ces passages pour répondre avec certitude."
                if language == "French" else "I couldn’t find enough evidence in these passages to answer reliably.",
                citations=[], used_model=True, raw=payload,
            )
        citation_ids = payload.get("citation_ids", [])
        if not isinstance(citation_ids, list):
            citation_ids = []
        # Also accept valid inline citations if the model omitted citation_ids.
        inline_ids = [
            item.strip() for group in re.findall(r"\[([^\]]+)\]", answer_text)
            for item in group.split(",") if item.strip()
        ]
        if any(not isinstance(item, str) or item not in allowed for item in [*citation_ids, *inline_ids]):
            raise ValueError("The model cited an unknown or unseen source")

        valid_ids = []
        for chunk_id in [*citation_ids, *inline_ids]:
            if chunk_id in allowed and chunk_id not in valid_ids:
                valid_ids.append(chunk_id)
        if not valid_ids:
            raise ValueError("The model returned no valid source citation")

        if payload.get("_require_quotes"):
            quotes = payload.get("evidence_quotes", {})
            normalize = lambda text: re.sub(r"\s+", " ", text).strip()
            # Small local models sometimes copy a real quote but attach the
            # neighbouring source ID. Reconcile only by exact source matching;
            # never accept a paraphrase or an ID outside the supplied context.
            verified_quote_ids = set()
            if isinstance(quotes, dict):
                for chunk_id, quote in quotes.items():
                    if (
                        chunk_id in allowed
                        and isinstance(quote, str)
                        and len(quote.strip()) >= 8
                        and normalize(quote) in normalize(allowed[chunk_id].text)
                    ):
                        verified_quote_ids.add(chunk_id)
            valid_ids = list(dict.fromkeys(
                [chunk_id for chunk_id in valid_ids if chunk_id in verified_quote_ids]
                + [chunk_id for chunk_id in verified_quote_ids if chunk_id in quotes]
            ))
            if not valid_ids:
                raise ValueError("No cited source contains an exact supporting quote")
        if not inline_ids:
            answer_text += " " + " ".join(f"[{item}]" for item in valid_ids)
        visualization = self._validate_visualization(payload.get("visualization"), allowed, valid_ids)
        for edge in visualization.edges:
            if edge.evidence_chunk_id not in valid_ids:
                valid_ids.append(edge.evidence_chunk_id)
        return Answer(
            text=answer_text,
            citations=[allowed[chunk_id] for chunk_id in valid_ids],
            visualization=visualization,
            used_model=True,
            raw=payload,
        )

    @staticmethod
    def _validate_visualization(
        raw: Any,
        allowed: dict[str, Passage],
        preferred_evidence_ids: list[str] | None = None,
    ) -> Visualization:
        if not isinstance(raw, dict):
            return Visualization()
        graph_type = str(raw.get("type", "none")).strip().lower()
        if graph_type not in {"relationship_graph", "timeline"}:
            return Visualization()

        nodes: list[GraphNode] = []
        node_ids: set[str] = set()
        for item in raw.get("nodes", []):
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("id", "")).strip()
            label = str(item.get("label", "")).strip()
            if not node_id or not label or node_id in node_ids:
                continue
            node_ids.add(node_id)
            nodes.append(GraphNode(node_id, label, str(item.get("kind", "character"))))

        edges: list[GraphEdge] = []
        for item in raw.get("edges", []):
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            evidence = str(item.get("evidence_chunk_id", "")).strip()
            label = str(item.get("label", "relation")).strip()
            # Some local models correctly extract the relation but copy the
            # schema placeholder instead of the source id. Repair only when
            # the answer itself has exactly one validated supporting passage.
            placeholder = bool(re.fullmatch(r"(?:chunk|source)(?:_id)?(?:_\d+)?", evidence, flags=re.IGNORECASE))
            if (
                evidence not in allowed
                and placeholder
                and preferred_evidence_ids
                and len(preferred_evidence_ids) == 1
            ):
                evidence = preferred_evidence_ids[0]
            if source in node_ids and target in node_ids and evidence in allowed:
                edges.append(GraphEdge(source, target, label, evidence))

        if not nodes or (graph_type == "relationship_graph" and not edges):
            return Visualization()
        return Visualization(graph_type, str(raw.get("title", "")).strip(), nodes, edges)

    @staticmethod
    def _visualization_from_graph(
        graph_result: GraphRetrievalResult,
        passages: list[Passage],
    ) -> Visualization:
        allowed = {passage.chunk_id: passage for passage in passages}
        nodes = {}
        edges = []
        for relationship in graph_result.relationships[:100]:
            evidence = relationship.evidence_chunk_id or ""
            if evidence not in allowed:
                continue
            work_id = allowed[evidence].work_id
            source_id = f"{work_id}:{relationship.source}"
            target_id = f"{work_id}:{relationship.target}"
            nodes[source_id] = GraphNode(source_id, relationship.source, relationship.source_kind)
            nodes[target_id] = GraphNode(target_id, relationship.target, relationship.target_kind)
            edges.append(GraphEdge(source_id, target_id, relationship.relation, evidence))
        if not edges:
            return Visualization()
        return Visualization(
            type="relationship_graph", title="Relations extraites du texte",
            nodes=list(nodes.values()), edges=edges,
        )

    @staticmethod
    def _extractive_fallback(question: str, passages: list[Passage], error: Exception) -> Answer:
        excerpts = []
        for passage in passages[:3]:
            text = passage.text.strip().replace("\n", " ")
            excerpts.append(f"- **{passage.citation_label}** : {text[:500]}…")
        return Answer(
            text=(
                ("Le modèle local n’a pas produit de réponse vérifiable. Voici les passages retrouvés :\n\n"
                 if LiteraryAssistant._answer_language(question) == "French"
                 else "The local model did not produce a verifiable answer. Here are the retrieved passages:\n\n")
                + "\n\n".join(excerpts)
            ),
            citations=passages[:3],
            used_model=False,
            raw={"error": str(error), "question": question},
        )


def visualization_to_dot(visualization: Visualization) -> str:
    """Convert a validated visualization to Graphviz DOT for Streamlit."""
    lines = [
        "digraph LiteraryGraph {",
        "rankdir=TB;",
        'graph [bgcolor="transparent", pad="0.35", nodesep="0.55", ranksep="0.75"];',
        'node [shape=box, style="rounded,filled", fillcolor="#F5E9D8", fontcolor="#151A22"];',
        'edge [color="#E8C07D", fontcolor="#F5E9D8", penwidth="1.5"];',
    ]
    for node in visualization.nodes:
        safe_id = json.dumps(node.id)
        safe_label = json.dumps(node.label, ensure_ascii=False)
        shape = "ellipse" if node.kind == "character" else "box"
        lines.append(f"{safe_id} [label={safe_label}, shape={shape}];")
    for edge in visualization.edges:
        safe_source = json.dumps(edge.source)
        safe_target = json.dumps(edge.target)
        safe_label = json.dumps(edge.label, ensure_ascii=False)
        lines.append(f"{safe_source} -> {safe_target} [label={safe_label}];")
    lines.append("}")
    return "\n".join(lines)
