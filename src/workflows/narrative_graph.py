from __future__ import annotations

import re
from typing import Any

from rag.models import Answer
from retrieval.spoiler_policy import SpoilerPolicy
from .state import NarrativeState

# Questions about how a book or a character ends. Asked before the reader got
# there, they are answered by a refusal, even with harmless early evidence.
ENDING_QUESTION = re.compile(
    r"\b(fin|finit|finir|termine|terminer|dénouement|denouement|dernier chapitre|meurt|mourir|décède"
    r"|end|ending|ends|finale|last chapter|die|dies|death)\b", re.IGNORECASE)
CHAPTER_REFERENCE = re.compile(r"\b(?:chapitre|chapter)\s+(\d+)", re.IGNORECASE)


class NarrativeWorkflow:
    """A bounded three-step workflow; LangGraph is optional at runtime."""

    def __init__(self, assistant) -> None:
        self.assistant = assistant
        self._graph = self._build_graph()

    def _guard_spoilers(self, state: NarrativeState) -> dict:
        max_chapter = state.get("max_chapter")
        question = state["question"]
        if max_chapter is None or not ENDING_QUESTION.search(question):
            return {"status": None}
        named = [int(n) for n in CHAPTER_REFERENCE.findall(question)]
        if named and max(named) <= max_chapter:
            return {"status": None}
        last = max((int(p.chapter) for p in self.assistant.index.passages
                    if p.work_id in state["work_ids"] and str(p.chapter).isdigit()), default=None)
        if last is None or max_chapter >= last:
            return {"status": None}
        french = self.assistant._answer_language(question) == "French"
        text = (f"Tu en es au chapitre {max_chapter} : cette question porte sur la suite du livre, je ne la dévoile donc pas. "
                f"Pose-moi une question sur les chapitres 1 à {max_chapter}, ou avance ta limite de lecture."
                if french else
                f"You are at chapter {max_chapter}: this question is about what happens later, so I won't reveal it. "
                f"Ask me about chapters 1 to {max_chapter}, or move your reading limit forward.")
        return {"status": "refused", "clarification": None,
                "answer": Answer(text=text, citations=[], used_model=False, raw={"spoiler_guard": True})}

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
        within_limit = all(SpoilerPolicy.allows(passage.chapter, max_chapter) for passage in (answer.citations if answer else []))
        if answer is not None and answer.raw.get("premise_rejected") and within_limit:
            return {"status": "answered", "clarification": None}
        if answer is not None and answer.citations and within_limit:
            return {"status": "answered", "clarification": None}
        return {"status": "needs_clarification", "clarification": "Je n’ai pas trouvé de preuve vérifiable dans les chapitres accessibles. Peux-tu préciser le passage ou le personnage ?"}

    @staticmethod
    def _route(state: NarrativeState) -> str:
        return "answered" if state.get("status") == "answered" else "clarify"

    @staticmethod
    def _route_guard(state: NarrativeState) -> str:
        return "refused" if state.get("status") == "refused" else "retrieve"

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
        graph.add_node("spoiler_guard", self._guard_spoilers)
        graph.add_node("retrieve", RunnableLambda(self._retrieve))
        graph.add_node("evidence_check", self._check_evidence)
        graph.add_node("clarify", self._clarify)
        graph.add_edge(START, "spoiler_guard")
        graph.add_conditional_edges("spoiler_guard", self._route_guard, {"refused": END, "retrieve": "retrieve"})
        graph.add_edge("retrieve", "evidence_check")
        graph.add_conditional_edges("evidence_check", self._route, {"answered": END, "clarify": "clarify"})
        graph.add_edge("clarify", END)
        return graph.compile()

    def invoke(self, state: NarrativeState) -> dict[str, Any]:
        if self._graph is not None:
            return self._graph.invoke(state, config={"recursion_limit": 8})
        result = {**state, **self._guard_spoilers(state)}
        if result.get("status") == "refused":
            return result
        result.update(self._retrieve(result))
        result.update(self._check_evidence(result))
        if result["status"] != "answered":
            result.update(self._clarify(result))
        return result
