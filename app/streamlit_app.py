from __future__ import annotations

import sys
import json
import hashlib
import logging
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for directory in (PROJECT_ROOT, SRC_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from catalog import GutendexClient, LibraryImporter  # noqa: E402
from catalog.gutendex import CatalogError  # noqa: E402
from rag.engine import LiteraryAssistant  # noqa: E402
from rag.models import GraphEdge, GraphNode, Visualization  # noqa: E402
from app.graph_component import render_relationship_graph  # noqa: E402
from app.ui_helpers import answer_message, render_message, friendly_error, export_markdown
from storage.local import ConversationStore, read_json, write_json


st.set_page_config(page_title="Literary Chat", page_icon="📚", layout="wide")

# Streamlit 1.40 can collapse an SVG returned by graphviz_chart to 0px when
# the SVG has only a viewBox (which is what the built-in renderer produces).
# Give relationship diagrams an explicit responsive drawing area.
st.markdown(
    """
    <style>
    :root {
        --primary-color: #416b5a;
        --lc-ground: #eef1ed;
        --lc-paper: #fcfcf8;
        --lc-ink: #26343a;
        --lc-muted: #65716f;
        --lc-green: #416b5a;
        --lc-lichen: #a8b58a;
        --lc-mist: #d7e0dc;
    }
    html, body, [class*="stApp"] { color: var(--lc-ink); font-family: "Aptos", "Segoe UI", sans-serif; }
    [data-testid="stAppViewContainer"] { background: var(--lc-ground); }
    [data-testid="stMain"] { background: var(--lc-ground); }
    [data-testid="stBottom"] { background: var(--lc-ground); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 1440px; padding-top: 2rem; padding-bottom: 4rem; }
    h1, h2, h3 { color: var(--lc-ink); letter-spacing: -0.035em; }
    h1 { font-size: clamp(2rem, 4vw, 3.1rem) !important; line-height: 1.08 !important; }
    h2, h3 { line-height: 1.2 !important; }
    [data-testid="stCaptionContainer"] { color: var(--lc-muted); }

    [data-testid="stSidebar"] { background: #e4e9e4; border-right: 1px solid #d0d9d3; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
    [data-testid="stSidebar"] h2 { font-size: 1.15rem; letter-spacing: -0.02em; }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] {
        background: var(--lc-green) !important; color: #fff !important; border-radius: 6px;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(252, 252, 248, .76); border: 1px solid #d4ddd7; border-radius: 12px;
    }

    /* Navigation between reading spaces stays in one compact ribbon. */
    [data-testid="stRadio"] [role="radiogroup"] {
        display: flex; flex-wrap: wrap; gap: .35rem; padding: .35rem;
        background: #e0e7e2; border: 1px solid #d1dbd4; border-radius: 13px;
        width: fit-content; max-width: 100%;
    }
    [data-testid="stRadio"] [role="radiogroup"] > label {
        margin: 0 !important; padding: .38rem .75rem; border-radius: 9px; color: #53625d;
        transition: background-color .16s ease, color .16s ease;
    }
    [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
        background: var(--lc-paper); color: var(--lc-green);
        box-shadow: 0 1px 2px rgba(38, 52, 58, .08); font-weight: 650;
    }
    [data-testid="stRadio"] [role="radiogroup"] > label > div:first-child { display: none; }

    [data-testid="stChatMessage"] {
        background: var(--lc-paper); border: 1px solid #dce3de; border-radius: 14px;
        padding: 1rem 1.2rem; margin-bottom: .85rem; box-shadow: 0 2px 8px rgba(38, 52, 58, .035);
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { line-height: 1.72; }
    [data-testid="stChatMessage"] [data-testid="stExpander"] {
        background: #f1f4ef; border: 0; border-left: 3px solid var(--lc-lichen);
        border-radius: 0 9px 9px 0; margin: .55rem 0;
    }
    [data-testid="stChatMessage"] [data-testid="stExpander"] p {
        font-family: Georgia, "Times New Roman", serif; line-height: 1.8;
    }
    [data-testid="stChatInput"] textarea { background: var(--lc-paper); border-color: #cbd6cf; border-radius: 13px; }
    [data-testid="stChatInput"] textarea:focus { border-color: var(--lc-green); }
    [data-testid="stSlider"] [role="slider"] { background-color: var(--lc-green) !important; }
    button[kind="primary"], [data-testid="stFormSubmitButton"] button {
        background: var(--lc-green) !important; border-color: var(--lc-green) !important; color: #fff !important;
    }
    button[kind="secondary"] { border-color: #cad5ce !important; color: var(--lc-ink) !important; background: var(--lc-paper) !important; }
    button { border-radius: 9px !important; }
    :focus-visible { outline: 3px solid #788f61 !important; outline-offset: 2px; }
    [data-testid="stMetric"] { background: var(--lc-paper); border: 1px solid #dce3de; border-radius: 12px; padding: .9rem 1rem; }
    [data-testid="stAlert"] { border-radius: 11px; }

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
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding: 1.25rem 1rem 3rem; }
        [data-testid="stRadio"] [role="radiogroup"] { width: 100%; gap: .15rem; }
        [data-testid="stRadio"] [role="radiogroup"] > label { padding: .38rem .5rem; }
        [data-testid="stChatMessage"] { padding: .8rem .9rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; }
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


@st.cache_data(ttl=60, show_spinner=False)
def get_graph_data(work_ids: tuple[str, ...]):
    graph_service = get_assistant().hybrid_retriever.graph
    return graph_service.explore(list(work_ids)) if graph_service else None



try:
    assistant = get_assistant()
except (ValueError, OSError) as error:
    st.error(friendly_error(error))
    st.stop()
catalog_client = get_catalog_client()
library_importer = LibraryImporter(PROJECT_ROOT / "data")
conversation_store = ConversationStore(PROJECT_ROOT / "data/library/conversations")
progress_path = PROJECT_ROOT / "data/library/reading_progress.json"
saved_progress = read_json(progress_path, {})
works = assistant.index.works
if "messages" not in st.session_state:
    st.session_state.messages = []
pending_books = st.session_state.pop("pending_books", None)
if pending_books is not None:
    st.session_state.pop("selected_books", None)
elif "selected_books" in st.session_state and not set(st.session_state.selected_books) <= set(works):
    st.session_state.pop("selected_books", None)


def refresh_library():
    assistant.hybrid_retriever.graph.close()
    get_assistant.clear()
    get_graph_data.clear()
    st.rerun()


def prepare_import(work_id):
    get_assistant.clear()
    get_graph_data.clear()
    current = get_assistant()
    st.session_state.pending_books = [work_id]
    semantic = current.hybrid_retriever.semantic
    if semantic:
        try:
            with st.status("Création de l’index sémantique…", expanded=True) as status:
                progress = st.progress(0.0)
                count = semantic.prepare(progress=lambda done, total: progress.progress(done / max(1, total)))
                status.update(label=f"Prêt · {count} passages indexés", state="complete", expanded=False)
        except Exception as error:
            st.session_state.catalog_warning = friendly_error(error) + " La recherche lexicale reste disponible."


st.title("📚 Literary Chat")
st.caption("Pose tes questions au fil de ta lecture. Chaque réponse s’appuie sur les passages du livre, sans quitter cette machine.")
if warning := st.session_state.pop("catalog_warning", None):
    st.warning(warning)

with st.sidebar:
    st.header("Bibliothèque")
    book_selector = dict(
        label="Romans consultés",
        options=list(works),
        key="selected_books",
        format_func=lambda work_id: f"{works[work_id].title} — {works[work_id].author}",
    )
    if "selected_books" in st.session_state:
        selected_works = st.multiselect(**book_selector)
    else:
        initial_books = ([work_id for work_id in pending_books if work_id in works]
                         if pending_books is not None else list(works)[:1])
        selected_works = st.multiselect(**book_selector, default=initial_books)
    top_k = st.slider("Passages analysés", min_value=3, max_value=8, value=4)
    max_chapter_options = [
        int(passage.chapter)
        for passage in assistant.index.passages
        if passage.work_id in selected_works and str(passage.chapter).isdigit()
    ]
    max_available_chapter = max(max_chapter_options, default=1)
    spoiler_setting = st.selectbox(
        "Protection contre les spoilers",
        ["Aucune restriction", "Jusqu’à un chapitre", "Livre terminé"],
    )
    spoiler_chapter = None
    if spoiler_setting == "Jusqu’à un chapitre":
        spoiler_chapter = st.number_input(
            "Je suis au chapitre…",
            min_value=1,
            max_value=max_available_chapter,
            value=min(saved_progress.get(selected_works[0], 1), max_available_chapter) if len(selected_works) == 1 else 1,
            key=f"reading_chapter_{'_'.join(selected_works)}",
        )
    if spoiler_chapter is not None and st.button("Enregistrer ma progression"):
        for work_id in selected_works:
            saved_progress[work_id] = int(spoiler_chapter)
        write_json(progress_path, saved_progress)
        st.success("Progression enregistrée.")
    answer_mode = st.selectbox(
        "Mode de lecture",
        ["Ask", "Explain", "Analyze", "Summarize", "Characters", "Quotes / Search", "Compare"],
        format_func=lambda mode: {"Ask": "Question", "Explain": "Expliquer", "Analyze": "Analyser",
                                  "Summarize": "Résumer", "Characters": "Personnages",
                                  "Quotes / Search": "Citations et recherche", "Compare": "Comparer"}[mode],
    )
    st.caption(f"Modèle local : `{assistant.model}`")
    if assistant.config.embedding_model:
        st.caption(f"Recherche sémantique : `{assistant.config.embedding_model}`")
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
                if cover := book.formats.get("image/jpeg"):
                    st.image(cover, width=70)
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
                        prepare_import(installed.work_id)
                        st.session_state.catalog_results = []
                        st.session_state.catalog_notice = f"« {installed.title} » est prêt dans la bibliothèque."
                        st.rerun()
                    except (CatalogError, ValueError, OSError) as exc:
                        st.error(str(exc))
                st.divider()

        st.markdown("**Importer votre propre exemplaire**")
        st.caption("EPUB, TXT, Markdown, PDF texte ou DOCX · 50 Mo maximum")
        upload = st.file_uploader(
            "Fichier du livre",
            type=["epub", "txt", "md", "pdf", "docx"],
            key="local_book_upload",
        )
        upload_title = st.text_input(
            "Titre",
            value=Path(upload.name).stem if upload else "",
            key="local_book_title",
        )
        upload_author = st.text_input("Auteur", key="local_book_author")
        upload_language = st.selectbox(
            "Langue du texte",
            options=["", "fr", "en", "es", "de", "it", "pt"],
            format_func=lambda value: value.upper() if value else "Non précisée",
            key="local_book_language",
        )
        if st.button(
            "Indexer ce fichier",
            disabled=upload is None,
            use_container_width=True,
        ):
            try:
                with st.spinner(f"Indexation de « {upload_title} »…"):
                    installed = library_importer.install_upload(
                        content=upload.getvalue(),
                        filename=upload.name,
                        title=upload_title,
                        author=upload_author,
                        language=upload_language,
                    )
                prepare_import(installed.work_id)
                st.session_state.catalog_notice = f"« {installed.title} » est prêt dans la bibliothèque."
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))


selected_passages = [
    p for p in assistant.index.passages if p.work_id in selected_works
    and (spoiler_chapter is None or (str(p.chapter).isdigit() and int(p.chapter) <= spoiler_chapter))
]
revision = hashlib.sha256("".join(p.chunk_id + p.text for p in selected_passages).encode()).hexdigest()
scope = json.dumps({"works": sorted(selected_works), "chapter": spoiler_chapter, "revision": revision}, sort_keys=True)
if st.session_state.get("conversation_scope") != scope:
    st.session_state.messages = []
    st.session_state.conversation_id = None
    st.session_state.pop("search_results", None)
    st.session_state.conversation_scope = scope

with st.sidebar:
    st.divider()
    if st.button("Nouvelle discussion", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()
    discussions = conversation_store.list(scope)
    if discussions:
        selected_discussion = st.selectbox("Discussions enregistrées", discussions, format_func=lambda item: item["title"])
        if st.button("Ouvrir la discussion"):
            st.session_state.messages = conversation_store.load(selected_discussion["id"], scope)
            st.session_state.conversation_id = selected_discussion["id"]
            st.rerun()
        if st.button("Archiver cette discussion"):
            conversation_store.archive(selected_discussion["id"])
            if st.session_state.get("conversation_id") == selected_discussion["id"]:
                st.session_state.messages = []
                st.session_state.conversation_id = None
            st.rerun()
    if st.session_state.messages:
        st.download_button("Exporter en Markdown", export_markdown(st.session_state.messages),
                           file_name="literary-chat.md", mime="text/markdown", use_container_width=True)
    archived = library_importer.archived_ids()
    if archived:
        records = {item["work_id"]: item for item in library_importer.records()}
        restore_id = st.selectbox("Livres archivés", sorted(archived), format_func=lambda value: records.get(value, {}).get("title", value))
        if st.button("Restaurer ce livre"):
            library_importer.set_archived(restore_id, False)
            refresh_library()

section = st.radio("Explorer", ["Chat", "Book", "Chapters", "Search", "Characters", "Knowledge Graph", "Insights"],
                   horizontal=True, label_visibility="collapsed",
                   format_func=lambda item: {"Chat": "Discussion", "Book": "Livre", "Chapters": "Chapitres",
                                             "Search": "Recherche", "Characters": "Personnages",
                                             "Knowledge Graph": "Graphe", "Insights": "Aperçu"}[item])
if not selected_works:
    st.info("Ajoute ou sélectionne un livre dans la bibliothèque pour commencer.")

if section == "Chat":
    if not st.session_state.messages and selected_works:
        st.markdown("### Où en est ta lecture ?")
        suggestions = ["Quels personnages apparaissent dans ces passages ?",
                       "Résume les chapitres accessibles.", "Trouve les passages sur la solitude."]
        for column, suggestion in zip(st.columns(3), suggestions):
            if column.button(suggestion, use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()
    for message in st.session_state.messages:
        render_message(message, assistant.index, spoiler_chapter)
    question = st.chat_input("Pose une question en français ou en anglais…", disabled=not selected_works)
    question = question or st.session_state.pop("pending_question", None)
    if question and selected_works:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-12:]]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        try:
            with st.status("Recherche dans le texte et vérification des sources…", expanded=False) as status:
                answer = assistant.answer(question, work_ids=selected_works, top_k=top_k, history=history,
                                          max_chapter=spoiler_chapter, mode=answer_mode)
                status.update(label="Réponse prête", state="complete")
            message = answer_message(answer)
            st.session_state.messages.append(message)
            render_message(message, assistant.index, spoiler_chapter)
            if error := answer.raw.get("error"):
                st.warning(friendly_error(RuntimeError(error)))
        except Exception as error:
            logging.exception("Chat request failed")
            st.error(friendly_error(error))
            st.session_state.messages.append({"role": "assistant", "content": friendly_error(error)})
        st.session_state.conversation_id = conversation_store.save(
            scope, st.session_state.messages, st.session_state.get("conversation_id")
        )

elif section == "Book":
    records = {item["work_id"]: item for item in library_importer.records()}
    for work_id in selected_works:
        work = works[work_id]
        record = records.get(work_id, {})
        passages = [p for p in assistant.index.passages if p.work_id == work_id]
        with st.container(border=True):
            if record.get("cover_url"):
                st.image(record["cover_url"], width=110)
            st.subheader(work.title)
            st.write(f"{work.author} · {(work.language or 'langue inconnue').upper()}")
            st.caption(f"{len({p.chapter for p in passages})} chapitres détectés · {len(passages)} passages")
            st.write(f"Source : {record.get('source', 'locale')}")
            if record.get("imported_at"):
                st.caption(f"Import : {record['imported_at'][:10]} · {record.get('size_bytes', 0) / 1024:.0f} Ko")
            left, right = st.columns(2)
            if left.button("Réindexer le livre", key=f"reindex_{work_id}"):
                try:
                    with st.spinner("Nouvelle extraction du texte…"):
                        library_importer.reindex(work_id)
                    prepare_import(work_id)
                    refresh_library()
                except Exception as error:
                    st.error(friendly_error(error))
            if right.button("Retirer de la bibliothèque", key=f"archive_{work_id}"):
                library_importer.set_archived(work_id)
                st.session_state.catalog_notice = "Livre archivé. Ses fichiers sont conservés ; restauration possible dans la barre latérale."
                refresh_library()

elif section == "Chapters":
    if selected_works:
        chapter_work = st.selectbox("Livre", selected_works, format_func=lambda work_id: works[work_id].title)
        grouped = {}
        for passage in selected_passages:
            if passage.work_id == chapter_work:
                grouped.setdefault(str(passage.chapter), []).append(passage)
        chapter = st.selectbox("Chapitre", sorted(grouped, key=lambda item: int(item) if item.isdigit() else 10**9)) if grouped else None
        if chapter:
            for passage in grouped[chapter]:
                with st.expander(passage.citation_label):
                    st.write(passage.text)
        st.caption("Seuls les chapitres autorisés par ta progression sont affichés. Les numéros sont détectés dans le fichier importé.")

elif section == "Search":
    query = st.text_input("Chercher une idée, un nom ou une citation")
    method = st.selectbox("Méthode", ["Hybrid", "BM25", "Sémantique"])
    if st.button("Rechercher", disabled=not query.strip() or not selected_works):
        try:
            with st.spinner("Recherche…"):
                if method == "BM25":
                    results = assistant.index.search(query, selected_works, top_k, spoiler_chapter)
                elif method == "Sémantique":
                    if not assistant.hybrid_retriever.semantic:
                        raise ValueError("La recherche sémantique est désactivée dans la configuration.")
                    results = assistant.hybrid_retriever.semantic.retrieve(query, selected_works, top_k, spoiler_chapter)
                else:
                    results = assistant.retrieve(query, selected_works, top_k, max_chapter=spoiler_chapter)
                st.session_state.search_results = [p.chunk_id for p in results]
        except Exception as error:
            st.session_state.search_results = []
            st.error(friendly_error(error))
    results = [assistant.index.get_passage(chunk_id) for chunk_id in st.session_state.get("search_results", [])]
    if "search_results" in st.session_state and not results:
        st.info("Aucun passage pertinent.")
    for passage in results:
        if passage:
            with st.expander(passage.citation_label):
                st.write(passage.text)

elif section in {"Characters", "Knowledge Graph"}:
    graph_service = assistant.hybrid_retriever.graph
    local_graph = graph_service.local_store
    st.caption("Extraction locale progressive. Chaque relation affichée est reliée à une citation exacte ; vérifie le passage pour confirmer l’interprétation.")
    batch_size = st.number_input("Passages à extraire dans ce lot", 1, 100, 5)
    if st.button("Construire / compléter le graphe", disabled=not selected_works):
        try:
            with st.status("Extraction des entités et relations…", expanded=True) as status:
                progress = st.progress(0.0)
                result = local_graph.build(assistant.llm, assistant.model, selected_works, int(batch_size),
                                           spoiler_chapter, progress=lambda done, total: progress.progress(done / max(1, total)))
                get_graph_data.clear()
                status.update(label=f"{result['processed']} passages traités · {result['remaining']} restants", state="complete")
        except Exception as error:
            logging.exception("Graph extraction failed")
            st.error(friendly_error(error))
    raw = get_graph_data(tuple(sorted(selected_works))) if selected_works else None
    graph = assistant.supported_graph(raw, selected_works, spoiler_chapter) if raw else None
    if graph and graph.relationships:
        names = sorted({r.source for r in graph.relationships} | {r.target for r in graph.relationships})
        selected_entity = st.selectbox("Entité", ["Toutes", *names])
        depth = st.radio("Voisinage", [1, 2], horizontal=True)
        visible = graph.relationships
        if selected_entity != "Toutes":
            neighborhood = {selected_entity}
            for _ in range(depth):
                neighborhood |= {name for r in visible if r.source in neighborhood or r.target in neighborhood for name in (r.source, r.target)}
            visible = [r for r in visible if r.source in neighborhood and r.target in neighborhood]
        kinds = {r.source: r.source_kind for r in visible} | {r.target: r.target_kind for r in visible}
        if section == "Characters":
            characters = [name for name in sorted(kinds) if kinds[name] == "character"]
            for name in characters:
                relations = [r for r in visible if name in {r.source, r.target}]
                with st.expander(f"{name} · {len(relations)} relations"):
                    for relation in relations:
                        passage = assistant.index.get_passage(relation.evidence_chunk_id)
                        st.write(f"{relation.source} — {relation.relation} → {relation.target}")
                        st.caption(passage.citation_label)
                        st.write(relation.evidence)
        else:
            visualization = assistant._visualization_from_graph(
                type(graph)(relationships=visible), selected_passages
            )
            render_relationship_graph(visualization, {p.chunk_id: p.text for p in selected_passages}, height=620)
        if st.button("Synchroniser les relations vers Neo4j"):
            if graph_service.available:
                try:
                    count = local_graph.sync_neo4j(graph_service.driver, selected_works)
                    st.success(f"{count} relations synchronisées.")
                except Exception as error:
                    st.error(friendly_error(error))
            else:
                st.warning("Neo4j est indisponible. Le graphe local reste utilisable. Lancez le service décrit dans le README.")
    else:
        st.info("Aucune relation sourcée pour cette sélection. Construis un premier lot pour commencer.")

elif section == "Insights":
    for work_id in selected_works:
        passages = [p for p in selected_passages if p.work_id == work_id]
        st.metric(works[work_id].title, f"{len(passages)} passages accessibles",
                  f"{len({p.chapter for p in passages})} chapitres")
    semantic = assistant.hybrid_retriever.semantic
    st.caption(f"Embeddings : {assistant.config.embedding_model or 'désactivés'} · cache {'présent' if semantic and semantic.cache_path.exists() else 'à construire'}")
    if st.button("Préparer / réparer les embeddings", disabled=not selected_passages or semantic is None):
        try:
            with st.status("Indexation sémantique…", expanded=True) as status:
                progress = st.progress(0.0)
                count = semantic.prepare(progress=lambda done, total: progress.progress(done / max(1, total)))
                status.update(label=f"{count} passages prêts", state="complete")
        except Exception as error:
            st.error(friendly_error(error))
    if st.button("Vérifier les services locaux"):
        try:
            assistant.llm.client.list()
            st.success("Ollama répond.")
        except Exception as error:
            st.warning(friendly_error(error))
        if assistant.hybrid_retriever.graph.available:
            st.success("Neo4j répond.")
        else:
            st.info("Neo4j est désactivé ou indisponible ; le graphe local est disponible.")
