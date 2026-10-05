from __future__ import annotations

from dataclasses import asdict

import streamlit as st

from app.graph_component import render_relationship_graph
from rag.models import GraphEdge, GraphNode, Visualization


def friendly_error(error: Exception) -> str:
    text = str(error).casefold()
    if any(word in text for word in ("connection", "connect", "refused", "timeout", "timed out")):
        return "Le service local ne répond pas. Lancez Ollama, puis réessayez. Les textes déjà importés restent consultables."
    if "not found" in text or "pull" in text:
        return "Un modèle Ollama est absent. Vérifiez les modèles dans les paramètres et installez-les avec ollama pull."
    if isinstance(error, (ValueError, OSError)):
        return str(error)
    return "L’opération a échoué. Vos données enregistrées sont conservées ; consultez le terminal pour le diagnostic."


def answer_message(answer) -> dict:
    return {
        "role": "assistant", "content": answer.text,
        "citations": [{"id": p.chunk_id, "label": p.citation_label, "text": p.text} for p in answer.citations],
        "visualization": asdict(answer.visualization),
        "elapsed_ms": answer.retrieval_ms + answer.generation_ms,
    }


def render_message(message, index, max_chapter=None):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for citation in message.get("citations", []):
            with st.expander(citation["label"]):
                st.write(citation["text"])
                neighbors = index.get_neighbors(citation.get("id", ""))
                neighbors = [p for p in neighbors if max_chapter is None or (str(p.chapter).isdigit() and int(p.chapter) <= max_chapter)]
                if neighbors:
                    st.caption("Passages voisins")
                    for passage in neighbors:
                        st.caption(passage.citation_label)
                        st.write(passage.text)
        raw = message.get("visualization", {})
        if raw.get("nodes"):
            visualization = Visualization(
                type=raw.get("type", "relationship_graph"), title=raw.get("title", ""),
                nodes=[GraphNode(**node) for node in raw.get("nodes", [])],
                edges=[GraphEdge(**edge) for edge in raw.get("edges", [])],
            )
            render_relationship_graph(visualization, {p.get("id", ""): p["text"] for p in message.get("citations", [])})
        if message.get("elapsed_ms"):
            st.caption(f"Réponse produite en {message['elapsed_ms'] / 1000:.1f} s")


def export_markdown(messages: list[dict]) -> str:
    parts = ["# Literary Chat\n"]
    for message in messages:
        parts.append(f"## {'Question' if message['role'] == 'user' else 'Réponse'}\n\n{message['content']}\n")
        for citation in message.get("citations", []):
            parts.append(f"### {citation['label']}\n\n> " + citation["text"].replace("\n", "\n> ") + "\n")
    return "\n".join(parts)
