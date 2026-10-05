from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for directory in (PROJECT_ROOT, SRC_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from catalog import GutendexClient, LibraryImporter  # noqa: E402
from catalog.gutendex import CatalogError  # noqa: E402
from rag.engine import LiteraryAssistant  # noqa: E402
from storage.local import ConversationStore, read_json, write_json  # noqa: E402


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    work_ids: list[str] = Field(min_length=1, max_length=10)
    top_k: int = Field(default=4, ge=3, le=8)
    max_chapter: int | None = Field(default=None, ge=1)
    mode: str = "Ask"
    history: list[dict[str, str]] = Field(default_factory=list, max_length=12)
    conversation_id: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    work_ids: list[str] = Field(min_length=1, max_length=10)
    method: str = "Hybrid"
    top_k: int = Field(default=6, ge=1, le=20)
    max_chapter: int | None = Field(default=None, ge=1)


class GraphRequest(BaseModel):
    work_ids: list[str] = Field(min_length=1, max_length=10)
    limit: int = Field(default=5, ge=1, le=100)
    max_chapter: int | None = Field(default=None, ge=1)


class ProgressRequest(BaseModel):
    work_ids: list[str] = Field(min_length=1, max_length=10)
    chapter: int = Field(ge=1)


class InstallRequest(BaseModel):
    provider_id: int = Field(gt=0)


class ConversationRequest(BaseModel):
    work_ids: list[str] = Field(min_length=1, max_length=10)
    max_chapter: int | None = Field(default=None, ge=1)


@lru_cache(maxsize=1)
def get_assistant() -> LiteraryAssistant:
    return LiteraryAssistant(
        manifest_path=PROJECT_ROOT / "data/annotations/work_manifest.json",
        processed_dir=PROJECT_ROOT / "data/processed",
    )


catalog = GutendexClient()
library = LibraryImporter(PROJECT_ROOT / "data")
conversations = ConversationStore(PROJECT_ROOT / "data/library/conversations")
progress_path = PROJECT_ROOT / "data/library/reading_progress.json"

api = FastAPI(title="Literary Chat API", version="1.0.0")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


def assistant_or_503() -> LiteraryAssistant:
    try:
        return get_assistant()
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def ensure_work_ids(work_ids: list[str], assistant: LiteraryAssistant) -> None:
    unknown = [work_id for work_id in work_ids if work_id not in assistant.index.works]
    if unknown:
        raise HTTPException(status_code=404, detail=f"Livre introuvable : {unknown[0]}")


def passage_dict(passage) -> dict:
    return {
        "work_id": passage.work_id,
        "work_title": passage.work_title,
        "author": passage.author,
        "chapter": passage.chapter,
        "chunk_id": passage.chunk_id,
        "text": passage.text,
        "score": passage.score,
        "section": passage.section,
        "page": passage.page,
        "pages": list(passage.pages),
        "citation_label": passage.citation_label,
    }


def graph_dict(graph, assistant: LiteraryAssistant, work_ids: list[str], max_chapter: int | None) -> dict:
    valid = assistant.supported_graph(graph, work_ids, max_chapter)
    kinds: dict[str, str] = {}
    for edge in valid.relationships:
        kinds[edge.source] = edge.source_kind
        kinds[edge.target] = edge.target_kind
    return {
        "nodes": [{"id": name, "label": name, "kind": kinds.get(name, "character")} for name in sorted(kinds)],
        "edges": [
            {"source": edge.source, "target": edge.target, "label": edge.relation,
             "evidence_chunk_id": edge.evidence_chunk_id, "evidence": edge.evidence,
             "work_id": edge.work_id, "source_kind": edge.source_kind, "target_kind": edge.target_kind}
            for edge in valid.relationships
        ],
    }


def conversation_scope(assistant: LiteraryAssistant, work_ids: list[str], max_chapter: int | None) -> str:
    passages = [
        passage for passage in assistant.index.passages
        if passage.work_id in work_ids
        and (max_chapter is None or (str(passage.chapter).isdigit() and int(passage.chapter) <= max_chapter))
    ]
    revision = hashlib.sha256("".join(p.chunk_id + p.text for p in passages).encode()).hexdigest()
    return json.dumps({"works": sorted(work_ids), "chapter": max_chapter, "revision": revision}, sort_keys=True)


@api.get("/api/health")
def health() -> dict:
    try:
        assistant = assistant_or_503()
        ollama_ok = bool(assistant.llm.client.list())
        graph_ok = bool(assistant.hybrid_retriever.graph.available)
        return {"ok": True, "model": assistant.model, "ollama": ollama_ok, "neo4j": graph_ok}
    except Exception:
        return {"ok": False, "ollama": False, "neo4j": False}


@api.get("/api/library")
def get_library() -> dict:
    assistant = assistant_or_503()
    records = {item.get("work_id"): item for item in library.records()}
    archived = library.archived_ids()
    works = []
    for work_id, work in assistant.index.works.items():
        record = records.get(work_id, {})
        passages = [p for p in assistant.index.passages if p.work_id == work_id]
        chapters = sorted({int(p.chapter) for p in passages if str(p.chapter).isdigit()})
        works.append({
            "id": work_id, "title": work.title, "author": work.author,
            "language": work.language, "passage_count": len(passages), "chapters": chapters,
            "cover_url": record.get("cover_url", ""), "source": record.get("source", "Corpus local"),
            "archived": work_id in archived,
        })
    return {
        "works": works,
        "model": assistant.model,
        "embedding_model": assistant.config.embedding_model,
        "stats": assistant.index.stats(),
        "progress": read_json(progress_path, {}),
        "supported_uploads": sorted(library.SUPPORTED_UPLOAD_SUFFIXES),
        "max_upload_bytes": library.MAX_UPLOAD_BYTES,
    }


@api.get("/api/books/{work_id}/passages")
def get_passages(
    work_id: str,
    chapter: int | None = Query(default=None, ge=1),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=30, ge=1, le=100),
) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids([work_id], assistant)
    passages = [p for p in assistant.index.passages if p.work_id == work_id]
    if chapter is not None:
        passages = [p for p in passages if str(p.chapter) == str(chapter)]
    return {"total": len(passages), "passages": [passage_dict(p) for p in passages[offset:offset + limit]]}


@api.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids(request.work_ids, assistant)
    history = request.history[-12:]
    answer = assistant.answer(
        request.question.strip(), work_ids=request.work_ids, top_k=request.top_k,
        history=history, max_chapter=request.max_chapter, mode=request.mode,
    )
    result = {
        "role": "assistant", "content": answer.text,
        "citations": [passage_dict(p) for p in answer.citations],
        "visualization": asdict(answer.visualization),
        "elapsed_ms": answer.retrieval_ms + answer.generation_ms,
        "used_model": answer.used_model,
    }
    scope = conversation_scope(assistant, request.work_ids, request.max_chapter)
    messages = history + [{"role": "user", "content": request.question.strip()}, result]
    result["conversation_id"] = conversations.save(scope, messages, request.conversation_id)
    return result


@api.post("/api/search")
def search(request: SearchRequest) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids(request.work_ids, assistant)
    method = request.method.casefold()
    if method == "bm25":
        passages = assistant.index.search(request.query, request.work_ids, request.top_k, request.max_chapter)
    elif method in {"sémantique", "semantique", "semantic"}:
        semantic = assistant.hybrid_retriever.semantic
        if semantic is None:
            raise HTTPException(status_code=409, detail="La recherche sémantique est désactivée dans la configuration.")
        passages = semantic.retrieve(request.query, request.work_ids, request.top_k, request.max_chapter)
    else:
        passages = assistant.retrieve(request.query, request.work_ids, request.top_k, max_chapter=request.max_chapter)
    return {"passages": [passage_dict(p) for p in passages]}


@api.get("/api/graph")
def get_graph(
    work_ids: Annotated[list[str], Query(min_length=1, max_length=10)],
    max_chapter: int | None = Query(default=None, ge=1),
) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids(work_ids, assistant)
    raw = assistant.hybrid_retriever.graph.local_store.explore(work_ids)
    return graph_dict(raw, assistant, work_ids, max_chapter)


@api.post("/api/graph/build")
def build_graph(request: GraphRequest) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids(request.work_ids, assistant)
    result = assistant.hybrid_retriever.graph.local_store.build(
        assistant.llm, assistant.model, request.work_ids, request.limit, request.max_chapter,
    )
    raw = assistant.hybrid_retriever.graph.local_store.explore(request.work_ids)
    return {**result, **graph_dict(raw, assistant, request.work_ids, request.max_chapter)}


@api.post("/api/graph/sync")
def sync_graph(request: GraphRequest) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids(request.work_ids, assistant)
    graph = assistant.hybrid_retriever.graph
    if not graph.available or graph.driver is None:
        raise HTTPException(status_code=503, detail="Neo4j n’est pas disponible. Le graphe local reste utilisable.")
    count = graph.local_store.sync_neo4j(graph.driver, request.work_ids)
    return {"synced": count}


@api.get("/api/catalog/search")
def catalog_search(query: str = Query(min_length=2, max_length=200), language: str | None = None, page: int = Query(default=1, ge=1)) -> dict:
    try:
        result = catalog.search(query, language=language, page=page)
    except (CatalogError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "count": result.count, "page": result.page,
        "has_next": result.has_next, "has_previous": result.has_previous,
        "books": [{"provider_id": b.provider_id, "title": b.title, "author": b.author_display,
                   "languages": list(b.languages), "language_display": b.language_display,
                   "subjects": list(b.subjects[:4]), "summary": b.summaries[0] if b.summaries else "",
                   "cover_url": b.formats.get("image/jpeg", ""), "download_count": b.download_count}
                  for b in result.books],
    }


@api.post("/api/catalog/install")
def install_catalog_book(request: InstallRequest) -> dict:
    try:
        book = catalog.get_book(request.provider_id)
        installed = library.install(book, catalog.download(book))
        get_assistant.cache_clear()
        assistant = get_assistant()
        warning = None
        if assistant.hybrid_retriever.semantic is not None:
            try:
                assistant.hybrid_retriever.semantic.prepare()
            except Exception as exc:
                logging.warning("Semantic indexing skipped after catalog import: %s", exc)
                warning = "Le livre est indexé pour la recherche textuelle; le service d’embeddings est indisponible."
        return {"work_id": installed.work_id, "title": installed.title, "already_installed": installed.already_installed, "warning": warning}
    except (CatalogError, ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.post("/api/books/upload")
def upload_book(
    file: Annotated[UploadFile, File()],
    title: str = "",
    author: str = "",
    language: str = "",
) -> dict:
    if file.size is not None and file.size > library.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Le fichier dépasse la limite de 50 Mo.")
    content = file.file.read(library.MAX_UPLOAD_BYTES + 1)
    if len(content) > library.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Le fichier dépasse la limite de 50 Mo.")
    try:
        installed = library.install_upload(content, file.filename or "livre.txt", title, author, language)
        get_assistant.cache_clear()
        assistant = get_assistant()
        warning = None
        if assistant.hybrid_retriever.semantic is not None:
            try:
                assistant.hybrid_retriever.semantic.prepare()
            except Exception as exc:
                logging.warning("Semantic indexing skipped after file import: %s", exc)
                warning = "Le livre est indexé pour la recherche textuelle; le service d’embeddings est indisponible."
        return {"work_id": installed.work_id, "title": installed.title, "already_installed": installed.already_installed, "warning": warning}
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.post("/api/library/archive/{work_id}")
def archive_book(work_id: str) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids([work_id], assistant)
    library.set_archived(work_id, True)
    return {"archived": True}


@api.post("/api/library/restore/{work_id}")
def restore_book(work_id: str) -> dict:
    library.set_archived(work_id, False)
    get_assistant.cache_clear()
    return {"archived": False}


@api.post("/api/library/reindex/{work_id}")
def reindex_book(work_id: str) -> dict:
    try:
        installed = library.reindex(work_id)
        get_assistant.cache_clear()
        return {"work_id": installed.work_id, "title": installed.title}
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.post("/api/progress")
def save_progress(request: ProgressRequest) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids(request.work_ids, assistant)
    progress = read_json(progress_path, {})
    for work_id in request.work_ids:
        progress[work_id] = request.chapter
    write_json(progress_path, progress)
    return {"progress": progress}


@api.post("/api/conversations")
def list_conversations(request: ConversationRequest) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids(request.work_ids, assistant)
    scope = conversation_scope(assistant, request.work_ids, request.max_chapter)
    return {"conversations": conversations.list(scope)}


@api.get("/api/conversations/{conversation_id}")
def load_conversation(
    conversation_id: str,
    work_ids: Annotated[list[str], Query(min_length=1, max_length=10)],
    max_chapter: int | None = Query(default=None, ge=1),
) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids(work_ids, assistant)
    scope = conversation_scope(assistant, work_ids, max_chapter)
    try:
        return {"id": conversation_id, "messages": conversations.load(conversation_id, scope)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.post("/api/conversations/{conversation_id}/archive")
def archive_conversation(conversation_id: str) -> dict:
    try:
        conversations.archive(conversation_id)
        return {"archived": True}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.post("/api/insights/embeddings")
def prepare_embeddings() -> dict:
    assistant = assistant_or_503()
    semantic = assistant.hybrid_retriever.semantic
    if semantic is None:
        raise HTTPException(status_code=409, detail="La recherche sémantique est désactivée dans la configuration.")
    count = semantic.prepare()
    return {"count": count, "embedding_model": assistant.config.embedding_model}


# The production frontend is built by `npm run build` in web/; API routes stay
# explicit above the catch-all so /api never leaks into the SPA fallback.
FRONTEND_DIST = PROJECT_ROOT / "web" / "dist"
if (FRONTEND_DIST / "index.html").is_file():
    api.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @api.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        candidate = FRONTEND_DIST / path
        if path and candidate.is_file() and FRONTEND_DIST in candidate.resolve().parents:
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")


app = api
