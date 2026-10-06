from __future__ import annotations

from typing import Any

from retrieval.spoiler_policy import SpoilerPolicy
from .state import NarrativeState


class NarrativeWorkflow:
    """A bounded two-branch workflow; LangGraph is optional at runtime."""

    def __init__(self, assistant) -> None:
        self.assistant = assistant
        self._graph = self._build_graph()

    def _retrieve(self, state: NarrativeState) -> dict:
        answer = self.assistant.answer(
            state["question"], work_ids=state["work_ids"],
            max_chapter=state.get("max_chapter"),
            top_k=state.get("top_k", 4), history=state.get("history", []),
            mode=state.get("mode", "Ask"), retrieval_query=state.get("retrieval_query"),
        )
        return {"answer": answer}

    @staticmethod
    def _check_evidence(state: NarrativeState) -> dict:
        answer = state.get("answer")
        max_chapter = state.get("max_chapter")
        valid = bool(answer and answer.citations) and all(
            SpoilerPolicy.allows(passage.chapter, max_chapter) for passage in answer.citations
        )
        if valid:
            return {"status": "answered", "clarification": None}
        return {"status": "needs_clarification", "clarification": "Je n’ai pas trouvé de preuve vérifiable dans les chapitres accessibles. Peux-tu préciser le passage ou le personnage ?"}

    @staticmethod
    def _route(state: NarrativeState) -> str:
        return "answered" if state.get("status") == "answered" else "clarify"

    @staticmethod
    def _clarify(state: NarrativeState) -> dict:
        answer = state.get("answer")
        if answer is not None:
            answer.text = state.get("clarification") or "Je ne trouve pas de preuve suffisante dans les chapitres accessibles."
            answer.citations = []
        return {"answer": answer}

    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            return None
        graph = StateGraph(NarrativeState)
        from langchain_core.runnables import RunnableLambda
        graph.add_node("retrieve", RunnableLambda(self._retrieve))
        graph.add_node("evidence_check", self._check_evidence)
        graph.add_node("clarify", self._clarify)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "evidence_check")
        graph.add_conditional_edges("evidence_check", self._route, {"answered": END, "clarify": "clarify"})
        graph.add_edge("clarify", END)
        return graph.compile()

    def invoke(self, state: NarrativeState) -> dict[str, Any]:
        if self._graph is not None:
            return self._graph.invoke(state, config={"recursion_limit": 8})
        result = {**state, **self._retrieve(state)}
        result.update(self._check_evidence(result))
        if result["status"] != "answered":
            result.update(self._clarify(result))
        return result
