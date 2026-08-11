from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag.engine import LiteraryAssistant, visualization_to_dot  # noqa: E402


st.set_page_config(page_title="Literary Chat", page_icon="📚", layout="wide")

# Streamlit 1.40 can collapse an SVG returned by graphviz_chart to 0px when
# the SVG has only a viewBox (which is what the built-in renderer produces).
# Give relationship diagrams an explicit responsive drawing area.
st.markdown(
    """
    <style>
    [data-testid="stGraphVizChart"] {
        width: 100% !important;
        min-height: 360px;
        padding: 0.5rem 0;
    }
    [data-testid="stGraphVizChart"] > svg {
        display: block;
        width: 100% !important;
        height: 340px !important;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_assistant() -> LiteraryAssistant:
    return LiteraryAssistant(
        manifest_path=PROJECT_ROOT / "data/annotations/work_manifest.json",
        processed_dir=PROJECT_ROOT / "data/processed",
    )


assistant = get_assistant()
works = assistant.index.works

st.title("📚 Literary Chat")
st.caption("Posez des questions sur vos romans. Recherche et génération entièrement locales.")

with st.sidebar:
    st.header("Bibliothèque")
    selected_works = st.multiselect(
        "Romans consultés",
        options=list(works),
        default=list(works),
        format_func=lambda work_id: f"{works[work_id].title} — {works[work_id].author}",
    )
    top_k = st.slider("Passages analysés", min_value=3, max_value=10, value=6)
    st.caption(f"Modèle local : `{assistant.model}`")
    for work_id, count in assistant.index.stats().items():
        st.caption(f"{works[work_id].title} : {count} passages")
    if st.button("Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("dot"):
            st.graphviz_chart(message["dot"], use_container_width=True)
        for citation in message.get("citations", []):
            with st.expander(citation["label"]):
                st.write(citation["text"])

question = st.chat_input("Ex. Pourquoi Gregor cache-t-il sa transformation à sa famille ?")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not selected_works:
            response_text = "Sélectionnez au moins un roman dans la barre latérale."
            st.warning(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:
            with st.spinner("Recherche dans les romans et génération locale…"):
                history = [
                    {"role": item["role"], "content": item["content"]}
                    for item in st.session_state.messages[:-1]
                ]
                answer = assistant.answer(
                    question,
                    work_ids=selected_works,
                    top_k=top_k,
                    history=history,
                )
            st.markdown(answer.text)
            dot = ""
            if answer.visualization.is_visible:
                if answer.visualization.title:
                    st.subheader(answer.visualization.title)
                dot = visualization_to_dot(answer.visualization)
                st.graphviz_chart(dot, use_container_width=True)

            citations = []
            if answer.citations:
                st.subheader("Passages sources")
                for passage in answer.citations:
                    citation = {"label": passage.citation_label, "text": passage.text}
                    citations.append(citation)
                    with st.expander(citation["label"]):
                        st.write(citation["text"])
            if not answer.used_model:
                st.info("Réponse de secours extractive : vérifiez qu’Ollama est lancé et que le modèle est installé.")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer.text, "citations": citations, "dot": dot}
            )
