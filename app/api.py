from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import sys
import httpx
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for directory in (PROJECT_ROOT, SRC_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from catalog import GutendexClient, LibraryImporter  # noqa: E402
from catalog.gutendex import CatalogError  # noqa: E402
from catalog.discovery import BookDiscovery  # noqa: E402
from rag.engine import LiteraryAssistant  # noqa: E402
from storage.local import ConversationStore, read_json, write_json  # noqa: E402
from retrieval.visual_retriever import LocalVisualIndex, MultimodalRetriever, TransformersClipProvider, VisualRecord  # noqa: E402
from vision import VisionAnalyzer  # noqa: E402
from vision.models import VisualObservation  # noqa: E402
from security.auth import AuthService  # noqa: E402
from security.access import BookAccess  # noqa: E402
from security.budget import current_budget, configured_budget, BudgetExceeded  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from time import perf_counter
from threading import BoundedSemaphore
from mcp_server.annotations import AnnotationService  # noqa: E402
from mcp_server.client import call_tool as call_mcp_tool  # noqa: E402
from retrieval.spoiler_policy import SpoilerPolicy  # noqa: E402
from workflows.narrative_graph import NarrativeWorkflow  # noqa: E402
from rag.prompts import QA_PROMPT_VERSION  # noqa: E402
from graph.identities import CharacterIdentities  # noqa: E402
from vision.ollama_vlm import PROMPT_VERSION as VISUAL_PROMPT_VERSION  # noqa: E402


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


class DiscoverRequest(BaseModel):
    query: str = Field(min_length=2, max_length=200)
    language: str = Field(default="", max_length=3)
    page: int = Field(default=1, ge=1, le=1000)
    source: Literal["all", "gutendex", "openlibrary"] = "all"
    use_model: bool = True


class MetadataRequest(BaseModel):
    provider: Literal["gutendex", "openlibrary"]
    provider_id: str = Field(min_length=1, max_length=30)


class ConversationRequest(BaseModel):
    work_ids: list[str] = Field(min_length=1, max_length=10)
    max_chapter: int | None = Field(default=None, ge=1)


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    password: str = Field(min_length=1, max_length=256)


class AnnotationPrepareRequest(BaseModel):
    work_id: str
    chapter: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=4000)
    evidence_ids: list[str] = Field(min_length=1, max_length=20)
    max_chapter: int | None = Field(default=None, ge=1)


class AnnotationSaveRequest(AnnotationPrepareRequest):
    approval_token: str = Field(min_length=1, max_length=1000)


class PassageCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    work_id: str
    work_title: str
    author: str
    chapter: int | str
    chunk_id: str
    text: str
    score: float
    section: str
    page: int | str | None
    pages: list[int]
    citation_label: str


class VisualCitationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["visual"] = "visual"
    work_id: str
    visual_id: str
    chapter: int | None
    page: int | None
    caption: str
    local_url: str


class GraphNodeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    label: str
    kind: str = "character"


class GraphEdgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    target: str
    label: str
    evidence_chunk_id: str


class VisualizationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str = "none"
    title: str = ""
    nodes: list[GraphNodeOutput] = Field(default_factory=list)
    edges: list[GraphEdgeOutput] = Field(default_factory=list)


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["assistant"] = "assistant"
    content: str
    status: Literal["answered", "needs_clarification", "refused", "action_pending"] = "answered"
    clarification_question: str | None = None
    citations: list[PassageCitation] = Field(default_factory=list)
    visual_citations: list[VisualCitationOutput] = Field(default_factory=list)
    visual_observation: VisualObservation | None = None
    visual_uncertainty: bool = False
    visual_analysis_available: bool = False
    visual_note: str | None = None
    visualization: VisualizationOutput = Field(default_factory=VisualizationOutput)
    elapsed_ms: int = 0
    used_model: bool = False
    conversation_id: str
    model_version: str
    vlm_version: str | None = None
    prompt_version: str
    visual_prompt_version: str | None = None
    reranker_version: str
    index_version: str


@lru_cache(maxsize=1)
def get_assistant() -> LiteraryAssistant:
    return LiteraryAssistant(
        manifest_path=PROJECT_ROOT / "data/annotations/work_manifest.json",
        processed_dir=PROJECT_ROOT / "data/processed",
    )


catalog = GutendexClient()
discovery = BookDiscovery(catalog)
library = LibraryImporter(PROJECT_ROOT / "data")
conversations = ConversationStore(PROJECT_ROOT / "data/library/conversations")
progress_path = PROJECT_ROOT / "data/library/reading_progress.json"
annotation_service = AnnotationService(PROJECT_ROOT / "data", os.getenv("APPROVAL_SECRET") or os.getenv("JWT_SECRET") or None)

api = FastAPI(title="NarrativeLens API", version="2.0.0")


class InferenceGuard(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.slot = BoundedSemaphore(max(1, int(os.getenv('MAX_CONCURRENT_INFERENCE', '1'))))

    async def dispatch(self, request, call_next):
        bounded = request.url.path in {'/api/chat', '/api/chat/multimodal', '/api/catalog/discover'} or request.url.path.startswith('/api/annotations')
        if not bounded:
            return await call_next(request)
        if not self.slot.acquire(blocking=False):
            return JSONResponse({'detail': 'Une demande est déjà en cours. Réessaie dans quelques instants.'}, status_code=429)
        budget = configured_budget()
        token = current_budget.set(budget)
        started = perf_counter()
        try:
            try:
                response = await call_next(request)
                budget.remaining()
            except BudgetExceeded as exc:
                response = JSONResponse({'detail': str(exc)}, status_code=504)
            response.headers['X-Model-Calls'] = str(budget.model_calls)
            response.headers['X-MCP-Calls'] = str(budget.mcp_calls)
            response.headers['Server-Timing'] = f'total;dur={(perf_counter()-started)*1000:.1f}'
            logging.info('request route=%s status=%s duration_ms=%d model_calls=%d mcp_calls=%d',
                         request.scope.get('route').path if request.scope.get('route') else '/api', response.status_code,
                         (perf_counter()-started)*1000, budget.model_calls, budget.mcp_calls)
            return response
        finally:
            current_budget.reset(token)
            self.slot.release()


api.add_middleware(InferenceGuard)
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


@lru_cache(maxsize=1)
def get_clip_provider():
    if os.getenv("VISUAL_EMBEDDINGS", "").casefold() != "clip":
        return None
    return TransformersClipProvider(
        os.getenv("VISUAL_EMBEDDING_MODEL", "sentence-transformers/clip-ViT-B-32-multilingual-v1"),
        os.getenv("VISUAL_IMAGE_MODEL", "clip-ViT-B-32"),
    )


def ensure_work_ids(work_ids: list[str], assistant: LiteraryAssistant, user_id: str = "local") -> None:
    unknown = [work_id for work_id in work_ids if work_id not in assistant.index.works]
    if unknown:
        raise HTTPException(status_code=404, detail=f"Livre introuvable : {unknown[0]}")
    if any(not BookAccess(PROJECT_ROOT / 'data').allows(user_id, work_id) for work_id in work_ids):
        raise HTTPException(status_code=403, detail="Accès à ce livre non autorisé.")


def _auth_service() -> AuthService:
    secret = os.getenv("JWT_SECRET", "")
    if len(secret) < 32:
        raise HTTPException(status_code=503, detail="AUTH_ENABLED requiert un JWT_SECRET d’au moins 32 caractères aléatoires.")
    return AuthService(PROJECT_ROOT / "data/library/users.json", secret)


def current_user_id(authorization: Annotated[str | None, Header()] = None) -> str:
    if os.getenv("AUTH_ENABLED", "false").casefold() not in {"1", "true", "yes"}:
        return "local"
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentification requise.")
    try:
        return _auth_service().verify(authorization[7:].strip())
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def _conversation_store(user_id: str) -> ConversationStore:
    if user_id == "local":
        return conversations
    user_key = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]
    return ConversationStore(PROJECT_ROOT / "data/library/users" / user_key / "conversations")


def _progress_file(user_id: str) -> Path:
    if user_id == "local":
        return progress_path
    user_key = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]
    return PROJECT_ROOT / "data/library/users" / user_key / "reading_progress.json"


def _reranker_version() -> str:
    provider = os.getenv("RERANKER_PROVIDER", "diversify").casefold()
    if provider == "trained":
        return os.getenv("RERANKER_PATH", "models/reranker")
    if provider == "cross_encoder":
        return os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    return "diversify"


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
    identities = {work: CharacterIdentities(assistant.hybrid_retriever.graph.local_store, work, max_chapter)
                  for work in work_ids}
    mappings = {work: resolver.mapping() for work, resolver in identities.items()}
    nodes = {}
    def add_node(work_id, name, kind, chunk_id):
        original = name
        mapping = mappings.get(work_id)
        label = name
        if kind == "character" and mapping is not None:
            entity = mapping.get((chunk_id, name))
            if entity is None:
                return None
            name, label, kind = entity["name"], entity["label"], entity["kind"]
        key = f"{work_id}:{name.casefold()}"
        passage = assistant.index.get_passage(chunk_id)
        node = nodes.setdefault(key, {"id": key, "label": label, "kind": kind,
                                      "work_id": work_id, "mentions": [], "aliases": [], "canonical_name": name})
        if original != label and original not in node["aliases"]:
            node["aliases"].append(original)
        if passage and not any(m["chunk_id"] == chunk_id for m in node["mentions"]):
            node["mentions"].append(passage_dict(passage))
        return key
    for node in valid.nodes:
        add_node(node["work_id"], node["name"], node["kind"], node["evidence_chunk_id"])
    edges = []
    for edge in valid.relationships:
        passage = assistant.index.get_passage(edge.evidence_chunk_id)
        work_id = passage.work_id
        source = add_node(work_id, edge.source, edge.source_kind, edge.evidence_chunk_id)
        target = add_node(work_id, edge.target, edge.target_kind, edge.evidence_chunk_id)
        if source is None or target is None:
            continue
        label = identities[work_id].data["labels"].get(
            CharacterIdentities.relation_key(edge.evidence_chunk_id, edge.source, edge.target, edge.relation), edge.relation)
        edges.append({"source": source, "target": target, "source_label": edge.source,
                      "target_label": edge.target, "label": label,
                      "source_mention": edge.source, "target_mention": edge.target,
                      "evidence_chunk_id": edge.evidence_chunk_id, "evidence": edge.evidence,
                      "work_id": work_id, "source_kind": edge.source_kind, "target_kind": edge.target_kind})
    aliases = {}
    for key, node in nodes.items():
        name = node["label"]
        if node["kind"] != "character" or mappings.get(node["work_id"]) is not None:
            continue
        without_article = re.sub(r"^(?:the|le|la|les)\s+", "", name, flags=re.IGNORECASE)
        candidates = [other for other, value in nodes.items()
                      if other != key and value["work_id"] == node["work_id"] and value["kind"] == "character"
                      and ((without_article != name and value["label"].casefold() == without_article.casefold())
                           or (len(name.split()) == 1 and name[:1].isupper()
                               and value["label"].startswith(name + " ")
                               and len(value["label"].split()) == 2
                               and value["label"].split()[1][:1].isupper()))]
        if len(candidates) == 1:
            aliases[key] = candidates[0]
    for old, canonical in aliases.items():
        target = nodes[canonical]
        target.setdefault("aliases", []).append(nodes[old]["label"])
        seen = {m["chunk_id"] for m in target["mentions"]}
        target["mentions"].extend(m for m in nodes[old]["mentions"] if m["chunk_id"] not in seen)
    for edge in edges:
        edge["source"] = aliases.get(edge["source"], edge["source"])
        edge["target"] = aliases.get(edge["target"], edge["target"])
    nodes = {key: value for key, value in nodes.items() if key not in aliases}
    edges = [e for e in edges if e["source"] != e["target"]]
    reviewed_works = set()
    for work_id, resolver in identities.items():
        reviewed = resolver.reviewed_relations()
        if reviewed is None:
            continue
        reviewed_works.add(work_id)
        edges = [edge for edge in edges if edge["work_id"] != work_id]
        for relation in reviewed:
            source_entity = resolver.review["entities"].get(relation["source_entity"])
            target_entity = resolver.review["entities"].get(relation["target_entity"])
            if not source_entity or not target_entity:
                continue
            source = f"{work_id}:{source_entity['name'].casefold()}"
            target = f"{work_id}:{target_entity['name'].casefold()}"
            if source in nodes and target in nodes and source != target:
                edges.append({"source": source, "target": target, "label": relation["label"],
                              "category": relation["category"], "work_id": work_id,
                              "evidence": relation["evidence"], "evidence_chunk_id": relation["evidence_chunk_id"]})
    # Explicit possessive kinship is factual evidence in its own right. This
    # recovers structural family links even when extraction chose a transient action.
    kinship = {"mother": "mère de", "father": "père de", "sister": "sœur de", "brother": "frère de"}
    family_seen = set()
    for mention in valid.nodes:
        match = re.fullmatch(r"(.+?)[’']s (mother|father|sister|brother)", mention["name"], re.I)
        if not match or mappings.get(mention["work_id"]) is None or mention["work_id"] in reviewed_works:
            continue
        source = add_node(mention["work_id"], mention["name"], mention["kind"], mention["evidence_chunk_id"])
        candidates = [n for n in nodes.values() if n["work_id"] == mention["work_id"] and n["kind"] == "character"
                      and match[1].casefold() in {a.casefold() for a in [n["canonical_name"], *n["aliases"]]}]
        if source is None or len(candidates) != 1 or candidates[0]["id"] == source:
            continue
        target = candidates[0]["id"]
        label = kinship[match[2].lower()]
        key = (source, target, label)
        if key in family_seen:
            continue
        family_seen.add(key)
        passage = assistant.index.get_passage(mention["evidence_chunk_id"])
        sentence = next((s.strip() for s in re.split(r"(?<=[.!?])\s+", passage.text) if mention["name"] in s), passage.text)
        edges.insert(0, {"source": source, "target": target, "label": label, "evidence": sentence,
                        "evidence_chunk_id": passage.chunk_id, "work_id": passage.work_id,
                        "source_kind": "character", "target_kind": "character", "category": "family"})
    for edge in edges:
        edge["source_label"] = nodes[edge["source"]]["label"]
        edge["target_label"] = nodes[edge["target"]]["label"]
        edge["source_kind"] = nodes[edge["source"]]["kind"]
        edge["target_kind"] = nodes[edge["target"]]["kind"]
    return {
        "nodes": sorted(nodes.values(), key=lambda n: (n["work_id"], n["label"].casefold())),
        "edges": edges,
        "coverage": assistant.hybrid_retriever.graph.local_store.coverage(work_ids, max_chapter),
        "identities": {"remaining": sum(r.remaining for r in identities.values()),
                       "unresolved_mentions": sum(sum(v is None for v in m.values()) for m in mappings.values() if m is not None)},
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
        try:
            response = httpx.get(f"{assistant.config.ollama_host.rstrip('/')}/api/tags", timeout=2.0)
            ollama_ok = response.is_success
        except httpx.HTTPError:
            ollama_ok = False
        graph_ok = bool(assistant.hybrid_retriever.graph.available)
        return {"ok": True, "model": assistant.model, "ollama": ollama_ok, "neo4j": graph_ok,
                "auth_enabled": os.getenv("AUTH_ENABLED", "false").casefold() in {"1", "true", "yes"}}
    except Exception:
        return {"ok": False, "ollama": False, "neo4j": False,
                "auth_enabled": os.getenv("AUTH_ENABLED", "false").casefold() in {"1", "true", "yes"}}


@api.post("/api/auth/register")
def register(request: Credentials) -> dict:
    if os.getenv("AUTH_ENABLED", "false").casefold() not in {"1", "true", "yes"}:
        raise HTTPException(status_code=409, detail="L’authentification n’est pas activée.")
    try:
        _auth_service().register(request.username, request.password)
        return {"created": True}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.post("/api/auth/login")
def login(request: Credentials) -> dict:
    try:
        return {"access_token": _auth_service().login(request.username, request.password), "token_type": "bearer"}
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@api.get("/api/library")
def get_library(user_id: str = Depends(current_user_id)) -> dict:
    assistant = assistant_or_503()
    records = {item.get("work_id"): item for item in library.records()}
    archived = library.archived_ids()
    works = []
    for work_id, work in assistant.index.works.items():
        if not BookAccess(PROJECT_ROOT / 'data').allows(user_id, work_id):
            continue
        record = records.get(work_id, {})
        passages = [p for p in assistant.index.passages if p.work_id == work_id]
        chapters = sorted({int(p.chapter) for p in passages if str(p.chapter).isdigit()})
        works.append({
            "id": work_id, "title": work.title, "author": work.author,
            "language": work.language, "passage_count": len(passages), "chapters": chapters,
            "cover_url": record.get("cover_url", ""), "source": record.get("source", "Corpus local"),
            "archived": work_id in archived,
            "summary": record.get("summary", ""), "subjects": record.get("subjects", []),
            "metadata_sources": record.get("metadata_sources", []), "cover_origin": record.get("cover_origin", ""),
            "first_publish_year": record.get("first_publish_year"),
        })
    # Archived books are deliberately absent from the retrieval index, but must
    # stay discoverable in the library so they can be restored after a restart.
    for work_id in sorted(archived - set(assistant.index.works)):
        if not BookAccess(PROJECT_ROOT / 'data').allows(user_id, work_id):
            continue
        record = records.get(work_id)
        if record:
            works.append({"id": work_id, "title": record.get("title", work_id),
                          "author": record.get("author", ""), "language": record.get("language", ""),
                          "passage_count": 0, "chapters": [], "cover_url": record.get("cover_url", ""),
                          "source": record.get("source", "Corpus local"), "archived": True})
    return {
        "works": works,
        "model": assistant.model,
        "embedding_model": assistant.config.embedding_model,
        "stats": {key: value for key, value in assistant.index.stats().items() if BookAccess(PROJECT_ROOT / 'data').allows(user_id, key)},
        "progress": {key: value for key, value in read_json(_progress_file(user_id), {}).items() if BookAccess(PROJECT_ROOT / 'data').allows(user_id, key)},
        "supported_uploads": sorted(library.SUPPORTED_UPLOAD_SUFFIXES),
        "max_upload_bytes": library.MAX_UPLOAD_BYTES,
    }


@api.get("/api/books/{work_id}/passages", dependencies=[Depends(current_user_id)])
def get_passages(
    work_id: str,
    chapter: int | None = Query(default=None, ge=1),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=30, ge=1, le=100),
    user_id: str = Depends(current_user_id),
) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids([work_id], assistant, user_id)
    passages = [p for p in assistant.index.passages if p.work_id == work_id]
    if chapter is not None:
        passages = [p for p in passages if str(p.chapter) == str(chapter)]
    return {"total": len(passages), "passages": [passage_dict(p) for p in passages[offset:offset + limit]]}


@api.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, user_id: str = Depends(current_user_id)) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids(request.work_ids, assistant, user_id)
    history = request.history[-12:]
    outcome = NarrativeWorkflow(assistant).invoke({
        "user_id": user_id, "question": request.question.strip(), "work_ids": request.work_ids,
        "top_k": request.top_k, "history": history, "max_chapter": request.max_chapter, "mode": request.mode,
    })
    answer = outcome["answer"]
    result = {
        "role": "assistant", "content": answer.text,
        "status": outcome.get("status", "answered"), "clarification_question": outcome.get("clarification"),
        "citations": [passage_dict(p) for p in answer.citations],
        "visualization": asdict(answer.visualization),
        "elapsed_ms": answer.retrieval_ms + answer.generation_ms,
        "used_model": answer.used_model,
        "model_version": assistant.model, "vlm_version": None,
        "visual_prompt_version": None,
        "prompt_version": QA_PROMPT_VERSION,
        "reranker_version": _reranker_version(),
        "index_version": hashlib.sha256(conversation_scope(assistant, request.work_ids, request.max_chapter).encode()).hexdigest()[:16],
    }
    scope = conversation_scope(assistant, request.work_ids, request.max_chapter)
    messages = history + [{"role": "user", "content": request.question.strip()}, result]
    result["conversation_id"] = _conversation_store(user_id).save(scope, messages, request.conversation_id)
    return result


@api.post("/api/chat/multimodal", response_model=ChatResponse)
def multimodal_chat(
    question: Annotated[str, Form(min_length=1, max_length=4000)],
    work_ids: Annotated[list[str], Form()],
    top_k: Annotated[int, Form(ge=3, le=8)] = 4,
    max_chapter: Annotated[int | None, Form(ge=1)] = None,
    mode: Annotated[str, Form()] = "Ask",
    history_json: Annotated[str, Form()] = "[]",
    conversation_id: Annotated[str | None, Form()] = None,
    file: UploadFile | None = File(default=None),
    user_id: str = Depends(current_user_id),
) -> dict:
    """Ask the existing literary RAG a question with an optional local image."""
    started = perf_counter()
    assistant = assistant_or_503()
    ensure_work_ids(work_ids, assistant, user_id)
    try:
        history_payload = json.loads(history_json)
        if not isinstance(history_payload, list):
            raise ValueError
        history = [{"role": str(item.get("role", "")), "content": str(item.get("content", ""))[:4000]}
                   for item in history_payload[-12:] if isinstance(item, dict) and item.get("role") in {"user", "assistant"}]
    except (ValueError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="Historique de conversation invalide.")
    observations = None
    visuals = []
    retrieval_query = question.strip()
    if file is not None:
        if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(status_code=415, detail="Formats d’image acceptés : JPG, PNG et WebP.")
        try:
            max_bytes = max(1024, min(int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024))), 50 * 1024 * 1024))
        except ValueError:
            max_bytes = 10 * 1024 * 1024
        payload = file.file.read(max_bytes + 1)
        if not payload:
            raise HTTPException(status_code=400, detail="L’image envoyée est vide.")
        if len(payload) > max_bytes:
            raise HTTPException(status_code=413, detail="L’image dépasse la taille maximale configurée.")
        signatures = {"image/jpeg": payload.startswith(b"\xff\xd8\xff"), "image/png": payload.startswith(b"\x89PNG\r\n\x1a\n"), "image/webp": len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP"}
        if not signatures.get(file.content_type, False):
            raise HTTPException(status_code=400, detail="Le contenu ne correspond pas au type d’image annoncé.")
        suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[file.content_type]
        with tempfile.NamedTemporaryFile(prefix="narrativelens-", suffix=suffix, delete=False) as temp:
            temp.write(payload)
            image_path = Path(temp.name)
        try:
            try:
                observations = VisionAnalyzer().analyze(image_path, question)
            except Exception as exc:
                logging.info("Vision provider unavailable for this request: %s", exc)
            try:
                clip_provider = get_clip_provider()
            except Exception as exc:
                logging.info("Visual embedding provider unavailable: %s", exc)
                clip_provider = None
            visual_index = LocalVisualIndex(PROJECT_ROOT / "data/processed", provider=clip_provider)
            try:
                visuals = visual_index.search(question, work_ids, top_k=top_k, max_chapter=max_chapter, image_path=image_path)
            except Exception as exc:
                logging.info("Visual retrieval unavailable for this request: %s", exc)
            terms = []
            if observations:
                # Image-derived text is retrieval input only; it never replaces
                # the user's question in the generation prompt.
                terms.extend([observations.ocr_text, observations.visual_description, *observations.visible_entities, *observations.objects])
            for visual in visuals:
                terms.extend([visual.caption, visual.ocr_text, visual.surrounding_text])
            retrieval_query = " ".join([question.strip(), *(term for term in terms if term)]).strip()
            outcome = NarrativeWorkflow(assistant).invoke({"user_id": user_id, "question": question.strip(), "retrieval_query": retrieval_query, "work_ids": work_ids,
                "top_k": top_k, "history": history, "max_chapter": max_chapter, "mode": mode,
                "image_analysis": observations.model_dump() if observations else None})
            answer = outcome["answer"]
        finally:
            image_path.unlink(missing_ok=True)
    else:
        outcome = NarrativeWorkflow(assistant).invoke({"user_id": user_id, "question": question.strip(), "retrieval_query": retrieval_query, "work_ids": work_ids,
            "top_k": top_k, "history": history, "max_chapter": max_chapter, "mode": mode})
        answer = outcome["answer"]

    if file is not None and visuals:
        fused = MultimodalRetriever.fuse(answer.citations, visuals, limit=max(top_k * 2, len(answer.citations) + len(visuals)))
        ordered_citations = [item for item in fused if not isinstance(item, VisualRecord)]
        visuals = [item for item in fused if isinstance(item, VisualRecord)]
    else:
        ordered_citations = answer.citations

    scope = conversation_scope(assistant, work_ids, max_chapter)
    result = {
        "role": "assistant", "content": answer.text,
        "status": outcome.get("status", "answered"), "clarification_question": outcome.get("clarification"),
        "citations": [passage_dict(p) for p in ordered_citations],
        "visual_citations": [{"type": "visual", "work_id": item.work_id, "visual_id": item.visual_id,
            "chapter": item.chapter, "page": item.page, "caption": item.caption,
            "local_url": f"/api/visuals/{item.visual_id}"} for item in visuals],
        "visual_observation": observations.model_dump() if observations else None,
        "visual_uncertainty": bool(observations and observations.uncertainties),
        "visual_analysis_available": observations is not None,
        "visual_note": None if observations else ("L’analyse d’image locale n’est pas disponible ; la question textuelle a tout de même été traitée." if file else None),
        "visualization": asdict(answer.visualization),
        "elapsed_ms": round((perf_counter() - started) * 1000),
        "used_model": answer.used_model,
        "model_version": assistant.model, "vlm_version": os.getenv("VLM_MODEL") if observations else None,
        "visual_prompt_version": VISUAL_PROMPT_VERSION if observations else None,
        "prompt_version": QA_PROMPT_VERSION,
        "reranker_version": _reranker_version(),
        "index_version": hashlib.sha256(conversation_scope(assistant, work_ids, max_chapter).encode()).hexdigest()[:16],
    }
    messages = history + [{"role": "user", "content": question.strip()}, result]
    result["conversation_id"] = _conversation_store(user_id).save(scope, messages, conversation_id)
    return result


@api.get("/api/visuals/{visual_id}")
def get_visual(visual_id: str, user_id: str = Depends(current_user_id)):
    if not re.fullmatch(r"[a-f0-9]{20}", visual_id):
        raise HTTPException(status_code=404, detail="Visuel introuvable.")
    visual_index = LocalVisualIndex(PROJECT_ROOT / "data/processed")
    record = next((item for item in visual_index.records if item.visual_id == visual_id
                   and BookAccess(PROJECT_ROOT / 'data').allows(user_id, item.work_id)), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Visuel introuvable.")
    path = Path(record.local_path).resolve()
    allowed_root = (PROJECT_ROOT / "data/library/visuals").resolve()
    if allowed_root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Visuel introuvable.")
    return FileResponse(path, media_type={".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}.get(path.suffix.casefold(), "application/octet-stream"))


def _validate_annotation_evidence(request: AnnotationPrepareRequest, user_id: str) -> None:
    assistant = assistant_or_503()
    ensure_work_ids([request.work_id], assistant, user_id)
    saved_progress = read_json(_progress_file(user_id), {}).get(request.work_id)
    limit = request.chapter
    if request.max_chapter is not None:
        limit = min(limit, request.max_chapter)
    if saved_progress is not None:
        limit = min(limit, int(saved_progress))
    for evidence_id in request.evidence_ids:
        passage = assistant.index.get_passage(evidence_id)
        if (passage is None or passage.work_id != request.work_id
                or not SpoilerPolicy.allows(passage.chapter, limit)):
            raise HTTPException(status_code=400, detail="Une preuve d’annotation est absente, hors livre ou après la limite de lecture.")


@api.post("/api/annotations/prepare")
def prepare_annotation(request: AnnotationPrepareRequest, user_id: str = Depends(current_user_id)) -> dict:
    _validate_annotation_evidence(request, user_id)
    try:
        if os.getenv('MCP_TRANSPORT', 'direct') == 'stdio':
            return call_mcp_tool('prepare_annotation', request.model_dump(exclude={'max_chapter'}),
                                 PROJECT_ROOT / 'data', user_id, annotation_service.secret.decode())
        return annotation_service.prepare(user_id, request.work_id, request.chapter, request.text, request.evidence_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.post("/api/annotations/save")
def save_annotation(request: AnnotationSaveRequest, user_id: str = Depends(current_user_id)) -> dict:
    _validate_annotation_evidence(request, user_id)
    try:
        if os.getenv('MCP_TRANSPORT', 'direct') == 'stdio':
            return call_mcp_tool('save_annotation', request.model_dump(exclude={'max_chapter'}),
                                 PROJECT_ROOT / 'data', user_id, annotation_service.secret.decode())
        return annotation_service.save(user_id, request.work_id, request.chapter, request.text, request.evidence_ids, request.approval_token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.get("/api/annotations")
def get_annotations(work_id: str | None = None, user_id: str = Depends(current_user_id)) -> dict:
    if work_id and os.getenv('MCP_TRANSPORT', 'direct') == 'stdio':
        if not BookAccess(PROJECT_ROOT / 'data').allows(user_id, work_id):
            raise HTTPException(status_code=403, detail='Accès à ce livre non autorisé.')
        return {'annotations': call_mcp_tool('get_user_annotations', {'work_id': work_id},
                                            PROJECT_ROOT / 'data', user_id, annotation_service.secret.decode())}
    return {"annotations": [item for item in annotation_service.list(user_id, work_id) if BookAccess(PROJECT_ROOT / "data").allows(user_id, item["work_id"])]}


@api.post("/api/search", dependencies=[Depends(current_user_id)])
def search(request: SearchRequest, user_id: str = Depends(current_user_id)) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids(request.work_ids, assistant, user_id)
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


@api.get("/api/graph", dependencies=[Depends(current_user_id)])
def get_graph(
    work_ids: Annotated[list[str], Query(min_length=1, max_length=10)],
    max_chapter: int | None = Query(default=None, ge=1),
    user_id: str = Depends(current_user_id),
) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids(work_ids, assistant, user_id)
    raw = assistant.hybrid_retriever.graph.local_store.explore(work_ids, max_chapter=max_chapter)
    return graph_dict(raw, assistant, work_ids, max_chapter)


@api.post("/api/graph/build", dependencies=[Depends(current_user_id)])
def build_graph(request: GraphRequest, user_id: str = Depends(current_user_id)) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids(request.work_ids, assistant, user_id)
    try:
        result = assistant.hybrid_retriever.graph.local_store.build(
            assistant.llm, assistant.model, request.work_ids, request.limit, request.max_chapter,
        )
        # Discover names only after the whole selected work has been extracted.
        # Chapter-restricted jobs must not classify characters using future text.
        if (result["remaining"] == 0
                and result["total"] == assistant.hybrid_retriever.graph.local_store.coverage(request.work_ids)["total"]):
            assistant.hybrid_retriever.graph.local_store.discover_characters(assistant.llm, assistant.model, request.work_ids)
        if result["remaining"] == 0:
            store = assistant.hybrid_retriever.graph.local_store
            with store._build_lock:
                for work_id in request.work_ids:
                    resolver = CharacterIdentities(store, work_id, request.max_chapter)
                    if resolver.remaining:
                        resolver.step(assistant.llm, assistant.model)
                        break
    except Exception as exc:
        logging.warning("Graph extraction failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="L’analyse du graphe a été interrompue. Vérifie Ollama puis reprends ; les passages terminés sont conservés.") from exc
    raw = assistant.hybrid_retriever.graph.local_store.explore(request.work_ids, max_chapter=request.max_chapter)
    return {**result, **graph_dict(raw, assistant, request.work_ids, request.max_chapter)}


@api.post("/api/graph/sync", dependencies=[Depends(current_user_id)])
def sync_graph(request: GraphRequest, user_id: str = Depends(current_user_id)) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids(request.work_ids, assistant, user_id)
    graph = assistant.hybrid_retriever.graph
    if not graph.available or graph.driver is None:
        raise HTTPException(status_code=503, detail="Neo4j n’est pas disponible. Le graphe local reste utilisable.")
    count = graph.local_store.sync_neo4j(graph.driver, request.work_ids)
    return {"synced": count}


@api.get("/api/catalog/search", dependencies=[Depends(current_user_id)])
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


@api.post("/api/catalog/discover", dependencies=[Depends(current_user_id)])
def discover_books(request: DiscoverRequest) -> dict:
    provider, model = None, None
    if request.use_model and request.page == 1 and len(request.query.split()) >= 5:
        assistant = assistant_or_503()
        provider, model = assistant.llm, assistant.model
    return discovery.search(request.query.strip(), request.language, request.page, request.source, provider, model)


@api.get("/api/catalog/details", dependencies=[Depends(current_user_id)])
def catalog_details(provider: Literal["gutendex", "openlibrary"], provider_id: str = Query(min_length=1, max_length=30)) -> dict:
    try:
        return discovery.details(provider, provider_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CatalogError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@api.post("/api/books/{work_id}/metadata", dependencies=[Depends(current_user_id)])
def attach_book_metadata(work_id: str, request: MetadataRequest, user_id: str = Depends(current_user_id)) -> dict:
    if not BookAccess(PROJECT_ROOT / 'data').allows(user_id, work_id):
        raise HTTPException(status_code=403, detail='Accès à ce livre non autorisé.')
    if not any(r["work_id"] == work_id for r in library.records()):
        raise HTTPException(status_code=404, detail="Livre inconnu.")
    try:
        library.attach_metadata(work_id, discovery.details(request.provider, request.provider_id))
        return {"work_id": work_id, "saved": True}
    except (ValueError, CatalogError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.post("/api/catalog/install", dependencies=[Depends(current_user_id)])
def install_catalog_book(request: InstallRequest, user_id: str = Depends(current_user_id)) -> dict:
    if not BookAccess(PROJECT_ROOT / 'data').allows(user_id, f'gutenberg_{request.provider_id}'):
        raise HTTPException(status_code=403, detail='Accès à ce livre non autorisé.')
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


@api.post("/api/books/upload", dependencies=[Depends(current_user_id)])
def upload_book(
    file: Annotated[UploadFile, File()],
    title: Annotated[str, Form(max_length=400)] = "",
    author: Annotated[str, Form(max_length=400)] = "",
    language: Annotated[str, Form(max_length=20)] = "",
    metadata_provider: Annotated[str, Form(max_length=30)] = "",
    metadata_id: Annotated[str, Form(max_length=30)] = "",
    user_id: str = Depends(current_user_id),
) -> dict:
    if file.size is not None and file.size > library.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Le fichier dépasse la limite de 50 Mo.")
    content = file.file.read(library.MAX_UPLOAD_BYTES + 1)
    if len(content) > library.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Le fichier dépasse la limite de 50 Mo.")
    upload_id = 'upload_' + hashlib.sha256(content).hexdigest()[:16]
    if not BookAccess(PROJECT_ROOT / 'data').allows(user_id, upload_id):
        raise HTTPException(status_code=403, detail='Accès à ce livre non autorisé.')
    try:
        metadata = None
        metadata_warning = None
        if metadata_provider or metadata_id:
            try:
                metadata = discovery.details(metadata_provider, metadata_id)
            except CatalogError:
                metadata_warning = "Le livre a été importé, mais sa fiche externe est indisponible. Tu peux la compléter depuis la bibliothèque."
        installed = library.install_upload(content, file.filename or "livre.txt", title, author, language, metadata=metadata)
        get_assistant.cache_clear()
        assistant = get_assistant()
        warning = metadata_warning
        if assistant.hybrid_retriever.semantic is not None:
            try:
                assistant.hybrid_retriever.semantic.prepare()
            except Exception as exc:
                logging.warning("Semantic indexing skipped after file import: %s", exc)
                warning = "Le livre est indexé pour la recherche textuelle; le service d’embeddings est indisponible."
        return {"work_id": installed.work_id, "title": installed.title, "already_installed": installed.already_installed, "warning": warning}
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.post("/api/library/archive/{work_id}", dependencies=[Depends(current_user_id)])
def archive_book(work_id: str, user_id: str = Depends(current_user_id)) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids([work_id], assistant, user_id)
    library.set_archived(work_id, True)
    get_assistant.cache_clear()
    return {"archived": True}


@api.post("/api/library/restore/{work_id}", dependencies=[Depends(current_user_id)])
def restore_book(work_id: str, user_id: str = Depends(current_user_id)) -> dict:
    if not BookAccess(PROJECT_ROOT / 'data').allows(user_id, work_id):
        raise HTTPException(status_code=403, detail='Accès à ce livre non autorisé.')
    try:
        library.set_archived(work_id, False)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Livre introuvable.") from exc
    get_assistant.cache_clear()
    return {"archived": False}


@api.post("/api/library/reindex/{work_id}", dependencies=[Depends(current_user_id)])
def reindex_book(work_id: str, user_id: str = Depends(current_user_id)) -> dict:
    if not BookAccess(PROJECT_ROOT / 'data').allows(user_id, work_id):
        raise HTTPException(status_code=403, detail='Accès à ce livre non autorisé.')
    try:
        installed = library.reindex(work_id)
        get_assistant.cache_clear()
        return {"work_id": installed.work_id, "title": installed.title}
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.post("/api/progress")
def save_progress(request: ProgressRequest, user_id: str = Depends(current_user_id)) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids(request.work_ids, assistant, user_id)
    target = _progress_file(user_id)
    progress = read_json(target, {})
    for work_id in request.work_ids:
        progress[work_id] = request.chapter
    write_json(target, progress)
    return {"progress": progress}


@api.post("/api/conversations")
def list_conversations(request: ConversationRequest, user_id: str = Depends(current_user_id)) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids(request.work_ids, assistant, user_id)
    scope = conversation_scope(assistant, request.work_ids, request.max_chapter)
    return {"conversations": _conversation_store(user_id).list(scope)}


@api.get("/api/conversations/{conversation_id}")
def load_conversation(
    conversation_id: str,
    work_ids: Annotated[list[str], Query(min_length=1, max_length=10)],
    max_chapter: int | None = Query(default=None, ge=1),
    user_id: str = Depends(current_user_id),
) -> dict:
    assistant = assistant_or_503()
    ensure_work_ids(work_ids, assistant, user_id)
    scope = conversation_scope(assistant, work_ids, max_chapter)
    try:
        return {"id": conversation_id, "messages": _conversation_store(user_id).load(conversation_id, scope)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.post("/api/conversations/{conversation_id}/archive")
def archive_conversation(conversation_id: str, user_id: str = Depends(current_user_id)) -> dict:
    try:
        _conversation_store(user_id).archive(conversation_id)
        return {"archived": True}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.post("/api/insights/embeddings", dependencies=[Depends(current_user_id)])
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
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Route API introuvable.")
        candidate = FRONTEND_DIST / path
        if path and candidate.is_file() and FRONTEND_DIST in candidate.resolve().parents:
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")


app = api
