from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catalog import GutendexClient, LibraryImporter  # noqa: E402
from catalog.gutendex import CatalogError  # noqa: E402
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


@st.cache_resource
def get_catalog_client() -> GutendexClient:
    return GutendexClient()


assistant = get_assistant()
catalog_client = get_catalog_client()
library_importer = LibraryImporter(PROJECT_ROOT / "data")
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

    if notice := st.session_state.pop("catalog_notice", None):
        st.success(notice)

    with st.expander("➕ Ajouter un livre", expanded=False):
        st.caption("Catalogue public Gutendex · Project Gutenberg")
        with st.form("catalog_search_form"):
            catalog_query = st.text_input(
                "Titre ou auteur",
                placeholder="Dostoevsky, Jane Austen, Candide…",
            )
            language_options = {
                "Toutes les langues": None,
                "Français": "fr",
                "Anglais": "en",
                "Espagnol": "es",
                "Allemand": "de",
                "Italien": "it",
                "Portugais": "pt",
            }
            language_label = st.selectbox("Langue", list(language_options))
            search_submitted = st.form_submit_button("Rechercher", use_container_width=True)

        if search_submitted:
            try:
                page = catalog_client.search(
                    catalog_query,
                    language=language_options[language_label],
                )
                st.session_state.catalog_results = list(page.books)
                st.session_state.catalog_total = page.count
                st.session_state.catalog_error = ""
            except (CatalogError, ValueError) as exc:
                st.session_state.catalog_results = []
                st.session_state.catalog_error = str(exc)

        if catalog_error := st.session_state.get("catalog_error"):
            st.error(catalog_error)

        catalog_results = st.session_state.get("catalog_results", [])
        if catalog_results:
            st.caption(f"{st.session_state.get('catalog_total', len(catalog_results))} résultat(s)")
            installed_ids = library_importer.installed_ids()
            for book in catalog_results:
                st.markdown(f"**{book.title}**")
                st.caption(
                    f"{book.author_display} · {book.language_display} · "
                    f"{book.download_count:,} téléchargements"
                )
                already_installed = book.work_id in installed_ids
                button_label = "✓ Déjà installé" if already_installed else "Télécharger et indexer"
                if st.button(
                    button_label,
                    key=f"install_catalog_{book.provider_id}",
                    disabled=already_installed,
                    use_container_width=True,
                ):
                    try:
                        with st.spinner(f"Indexation de « {book.title} »…"):
                            installed = library_importer.install(book, catalog_client.download(book))
                        get_assistant.clear()
                        st.session_state.catalog_results = []
                        st.session_state.catalog_notice = f"« {installed.title} » est prêt dans la bibliothèque."
                        st.rerun()
                    except (CatalogError, ValueError, OSError) as exc:
                        st.error(str(exc))
                st.divider()

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
